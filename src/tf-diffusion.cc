#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "mfem.hpp"
#include "util.h"

using namespace std;
using namespace mfem;

double SetInitalValues(const Vector &x);

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file = "../data/star.mesh";
  bool visualization = true;
  bool generate_mesh = true;
  int dim = 2;
  int ref_levels = 2;
  int order = 2;
  int vis_steps = 5;
  int max_iter = 100;
  double t_final = 1.0;
  double dt = 1.0e-2;
  double diffusion_constant = 1.0;
  double rel_tol = 1e-8;
  const char *device_config = "omp";

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&generate_mesh, "-genmesh", "--generate-mesh", "-no-genmesh",
                 "--no-generate-mesh",
                 "Generate Mesh or not. If not, mesh file is used.");
  args.AddOption(&dim, "-d", "--dimension",
                 "Dimension of the problem (1, 2, or 3).");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.AddOption(&max_iter, "-mi", "--maximal-iterations",
                 "Maximal number of iterations for CG solver.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step",
                 "Time step. Corresponds to h in publication.");
  args.AddOption(&diffusion_constant, "-k", "--kappa", "Diffusion constant.");
  args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                 "Relative tolerance of CGSolver.");
  args.AddOption(&device_config, "-dev", "--device",
                 "Device configuration string, see Device::Configure().");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Device device(device_config);
  device.Print();

  // 2. Either read or generate the mesh.
  Mesh *mesh = new Mesh();
  if (!generate_mesh) {
    mesh = new Mesh(mesh_file, 1, 1);
    dim = mesh->Dimension();
  } else {
    switch (dim) {
      case 1:
        *mesh = Mesh::MakeCartesian1D(10);
        break;
      case 2:
        *mesh = Mesh::MakeCartesian2D(10, 10, Element::Type::TRIANGLE);
        break;
      case 3:
        *mesh = Mesh::MakeCartesian3D(10, 10, 10, Element::Type::TETRAHEDRON);
        break;
      default:
        throw std::invalid_argument(
            "Dimension argument must be either 1, 2, or 3.");
        break;
    }
  }

  // 3. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  // 4. Read the expansion coefficients c_k, d_k, and c_/inf
  const std::string filename_alpha{"../data/alpha.npy"};
  const std::string filename_d{"../data/d.npy"};
  const std::string filename_c{"../data/c.npy"};
  const std::string filename_cinf{"../data/c_inf.npy"};
  const std::vector<double> alpha_vec = ReadNPYVector(filename_alpha);
  const std::vector<double> d = ReadNPYVector(filename_d);
  const std::vector<double> c = ReadNPYVector(filename_c);
  const std::vector<double> cinf = ReadNPYVector(filename_cinf);
  const double c_inf = cinf[0];

  // 5. Define the vector finite element space representing the current and the
  //    initial temperature, u_ref.
  H1_FECollection fe_coll(order, dim);
  FiniteElementSpace fespace(mesh, &fe_coll);

  int fe_size = fespace.GetTrueVSize();
  cout << "Number of Elements : " << mesh->GetNE() << endl;
  cout << "Number of DoFs     : " << fe_size << endl;
  cout << "Value of alpha     : " << alpha_vec[0] << endl;

  GridFunction u_gf(&fespace);

  // 6. Set the initial conditions for u. All boundaries are considered
  //    natural.
  FunctionCoefficient u_0(SetInitalValues);
  u_gf.ProjectCoefficient(u_0);
  Vector u;
  u_gf.GetTrueDofs(u);

  // 7. Save mesh, initial configuration, and initialize visualization.
  {
    ofstream omesh("tf-diffusion.mesh");
    omesh.precision(precision);
    mesh->Print(omesh);
    ofstream osol("tf-diffusion-init.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  // Remark: make sure you've started `glvis -mac` in a terminal.
  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    sout.open(vishost, visport);
    if (!sout) {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      cout << "Please run `./glvis -mac` to enable visualization server."
           << std::endl;
      visualization = false;
      cout << "GLVis visualization disabled.\n";
    } else {
      sout.precision(precision);
      sout << "solution\n" << *mesh << u_gf;
      sout << "pause\n";
      sout << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
    }
  }

  // 8. Compute the relevant coefficients of the fractal time integration.
  const std::vector<double> gamma = ComputeGamma(d, dt);
  const std::vector<double> beta_1 = ComputeBeta1(d, c, gamma, dt);
  const std::vector<double> beta_2 = ComputeBeta2(d, c, gamma, dt);
  const double beta = ComputeBeta(beta_1, beta_2, c_inf);

  // 9. Compute FE coefficient vectors and matrices

  // Boundary - Direchlet 0 on boudary.
  Array<int> boundary_dofs;
  fespace.GetBoundaryTrueDofs(boundary_dofs);

  // Mass matrix
  BilinearForm M(&fespace);
  SparseMatrix Mmat;
  M.AddDomainIntegrator(new MassIntegrator());
  M.SetAssemblyLevel(AssemblyLevel::LEGACY);
  M.Assemble();
  M.FormSystemMatrix(boundary_dofs, Mmat);

  // Stiffness matrix
  BilinearForm D(&fespace);
  SparseMatrix Dmat;
  ConstantCoefficient d_coeff{diffusion_constant};
  D.AddDomainIntegrator(new DiffusionIntegrator(d_coeff));
  D.SetAssemblyLevel(AssemblyLevel::LEGACY);
  D.Assemble();
  D.FormSystemMatrix(boundary_dofs, Dmat);

  // Matrix T that is (M + beta D). Add(...) creates on heat - needs to be
  // deleted manually at the end.
  SparseMatrix *Tmat;
  Tmat = Add(1.0, Mmat, beta, Dmat);

  // Conjugate Gradient solver to applying Tmat^{-1}
  CGSolver T_solver;
  DSmoother T_prec;
  T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(max_iter);
  T_solver.SetPrintLevel(0);
  T_solver.SetPreconditioner(T_prec);  // To be called before SetOperator.
  T_solver.SetOperator(*Tmat);

  // Vector M*u_0
  Vector Mu0{fe_size};
  Mmat.Mult(u, Mu0);

  // Vector corresponding to the modes M*u_k=w_k. Intialized to zero.
  std::vector<Vector> w_1;
  std::vector<Vector> w_2;
  for (size_t i = 0; i < d.size(); i++) {
    Vector tmp_1{fe_size};  // Size of DoF
    tmp_1 *= 0.0;           // Initialize to Zero
    w_1.push_back(tmp_1);
  }

  // Additional vector for u^{n+1}
  Vector u_new{u};

  // 10. Perform fractal-time-integration
  double t = 0.0;

  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    // Determine if we are in the last iteration.
    if (t + dt >= t_final - dt / 2) {
      last_step = true;
    }

    // Time integration
    // 10.1 scale w_1 with gamma
    for (size_t k = 0; k < w_1.size(); k++) {
      w_1[k] *= gamma[k];
    }
    // 10.2 Add up all vectors of RHS
    Vector rhs{Mu0};
    for (size_t k = 0; k < w_1.size(); k++) {
      rhs += w_1[k];
    }
    // 10.3 Compute u^{n+1} by solving a matrix equation
    T_solver.Mult(rhs, u_new);
    // 10.4 Update w_2
    for (size_t k = 0; k < w_1.size(); k++) {
      Vector tmp1{fe_size};
      Dmat.Mult(u_new, tmp1);
      tmp1 *= beta_1[k];
      w_1[k] -= tmp1;
      Dmat.Mult(u, tmp1);
      tmp1 *= beta_2[k];
      w_1[k] -= tmp1;
    }
    // 10.5 Update time
    t += dt;

    // Visualization
    if (last_step || (ti % vis_steps) == 0) {
      cout << "step " << ti << ", t = " << t << endl;

      u_gf.SetFromTrueDofs(u_new);
      if (visualization) {
        sout << "solution\n" << *mesh << u_gf << flush;
      }
    }

    // Swap pointers of u_new and u
    std::swap(u_new, u);
  }

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m ex16.mesh -g ex16-final.gf".
  {
    ofstream osol("ex16-final.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  // 10. Free the used memory.
  delete mesh;
  delete Tmat;

  return 0;
}

double SetInitalValues(const Vector &x) {
  int dimension{x.Size()};
  double result{1.0};
  const double pi = std::acos(-1);
  for (int i = 0; i < dimension; i++) {
    result *= std::sin(pi * x[i]);
  }
  return result;

  // // Alternative initial conditions below.

  // // Code snipped needed for both options below to compute distance.
  // int dimension{x.Size()};
  // double distance{0.0};
  // const double pi = std::acos(-1);
  // for (int i = 0; i < dimension; i++) {
  //   distance += std::pow(x[i] - 0.5, 2);
  // }
  // distance = std::sqrt(distance);

  // // Peak of 1 in a certain area in the middle
  // if (distance > 0.15) {
  //   return 0;
  // } else {
  //   return 1;
  // }

  // // Peak of 1 at a certain point in the middle decreasing linearly to the
  // // Surrounding.
  // if (distance < 0.2) {
  //   return 0.2 - distance;
  // } else {
  //   return 0;
  // }
}
