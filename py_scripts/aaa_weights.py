# coding: utf-8
#
# This script approximates the function z^{-\alpha} with the AAA algorithm. The
# approximation takes the shape c_\inf + sum_k \frac{c_k}{z+d_k}. The script
# computes the factors c_/inf, c_k, and d_k.
#
# Set up conda environment to run the script:
#   `conda env create -f py_scripts/conda_environment.yml`
#   `conda list | grep fractional`
#   `conda activate fractional`
#
# Get information on how to use the script:
#   `python py_scripts/aaa_weights.py --help`
#
# Run the script
#   `python py_scripts/aaa_weights.py`
#
# History:
#   Initial Version: Julius Schmid, Andreas Wagner (TUM)
#   Update         : Tobias Duswald (CERN, TUM)
#


from absl import app
from absl import flags
import numpy as np
from baryrat import aaa
from scipy.signal import zpk2tf
from scipy.signal import residue
from matplotlib import pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_float(
    "alpha", 0.5, r"Exponent alpha in z^{-\alpha} (Function to approximate)"
)
flags.DEFINE_float(
    "tau", 1e-2 / 128.0, "Lower bound considered in approximation."
)
flags.DEFINE_float("tmax", 1, "Upper bound considered in approximation.")
flags.DEFINE_float("tol", 1e-14, "Tolerance of AAA algorithm.")
flags.DEFINE_integer("npoints", 100, "Number of points for sampling.")
flags.DEFINE_boolean(
    "visualize", False, "Visualize the convergence of the approximation."
)
flags.DEFINE_boolean(
    "print", False, "Print out the coefficients c_k, d_k, and c_\inf."
)
flags.DEFINE_boolean(
    "save", True, "Save the coefficitents c_k, d_k, and c_\inf."
)


def CalculateWeights(alpha, tau, t_max, tol=1e-14):
    """
    Computes the weights c_\inf, c_k, and d_k for a given alpha in z^{-\alpha}.
    """
    tau = tau / 2.0
    t_max = 2 * t_max
    z = np.geomspace(tau, t_max, 100)

    z_alpha = np.power(z, alpha)

    r = aaa(z, z_alpha, tol=tol)

    coeff = np.dot(r.values, r.weights) / np.sum(r.weights)

    poles, _ = r.polres()
    zeros = r.zeros()

    # make sure that imaginary parts are neglegible:
    assert np.linalg.norm(np.imag(poles)) / np.linalg.norm(poles) < 1e-10
    assert np.linalg.norm(np.imag(zeros)) / np.linalg.norm(zeros) < 1e-10

    num_pol, denom_pol = zpk2tf(zeros, poles, coeff)

    # reverse order to get z^{-alpha} instead of z^{alpha}
    num_pol = np.array(list(num_pol)[::-1])
    denom_pol = np.array(list(denom_pol)[::-1])

    # make sure that imaginary parts are neglegible:
    assert np.linalg.norm(np.imag(num_pol)) / np.linalg.norm(num_pol) < 1e-10
    assert (
        np.linalg.norm(np.imag(denom_pol)) / np.linalg.norm(denom_pol) < 1e-10
    )

    # remove potential imaginary parts:
    num_pol = np.real(num_pol)
    denom_pol = np.real(denom_pol)

    # normalizing the denominator and nominator polynomial improves the
    # conditioning of the partial fraction expansion:
    prefactor = num_pol[0] / denom_pol[0]
    num_pol /= num_pol[0]
    denom_pol /= denom_pol[0]

    # do the partial fraction expansion
    c, d, c_infty = residue(num_pol, denom_pol)

    # reapply the prefactor and invert the d factor
    d = -d
    c = prefactor * c
    c_infty = prefactor * c_infty

    return c, d, c_infty


def main(argv):
    """
    Main function. Handles parameters, calls CalculateWeights, computes errors,
    saves results, and - optionally - visualizes the approximation.
    """

    # Parameters for sampling, AAA algorithm, and turning on and off optional
    # features
    alpha = FLAGS.alpha
    tau = FLAGS.tau
    t_max = FLAGS.tmax
    n_points = FLAGS.npoints
    tol = FLAGS.tol
    visualize = FLAGS.visualize
    print_coefficients = FLAGS.print
    save_coefficients = FLAGS.save

    # X points to evaluate function on
    t = np.linspace(1.0 / t_max, 1.0 / tau, n_points)

    # Function values z^{-\alpha}(t)
    function_values = np.power(t, -alpha)

    # Compute the relevant weights for approximation needed for FE code
    c, d, c_infty = CalculateWeights(alpha, tau, t_max, tol)
    c = c.astype("float64")
    d = d.astype("float64")
    c_infty = c_infty.astype("float64")

    # Define function to evaluate the computed Approximation
    def EvalPoly(t):
        return np.sum(c / (t + d)) + c_infty

    EvalPoly = np.vectorize(EvalPoly)

    # Compute the approximation
    approximation = EvalPoly(t)

    # Compute the error of the approximation
    error = np.abs(approximation - function_values)
    error_L1 = error.sum() / error.size
    error_L2 = np.sqrt(np.square(error).sum()) / error.size
    error_max = np.max(error)

    # Print approximation information
    print()
    print("Alpha                     :", alpha)
    print("L1 error of approximation :", error_L1)
    print("L2 error (RMS) of approx. :", error_L2)
    print("Maximum absolute error    :", error_max)
    print("No. c_k                   :", c.size)
    print("No. d_k                   :", d.size)
    print()

    # Print the coefficients to the command line
    if print_coefficients:
        print("c", repr(c))
        print("d", repr(d))
        print("c_infty", repr(c_infty))

    # Save the coefficients to "timefractional-diffusion/data"
    if save_coefficients:
        import os

        path = os.getcwd() + "/data"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save("data/alpha", np.array([alpha]).astype("float64"), False)
        np.save("data/c", c, False)
        np.save("data/d", d, False)
        np.save("data/c_inf", c_infty, False)

    # Visualize approximation
    if visualize:

        fig, axs = plt.subplots(2)
        fig.suptitle("Convergence")

        axs[0].plot(t, function_values, label=r"$z^{-\alpha}$")
        axs[0].plot(t, approximation, "--", label=r"AAA-Approx")
        axs[0].legend()
        axs[0].set_yscale("log")
        axs[0].set_ylabel("f(x)")
        axs[0].set_xscale("log")
        axs[0].grid(True)

        axs[1].plot(t, error, label="error")
        axs[1].legend()
        axs[1].set_yscale("log")
        axs[1].set_ylabel(r"$ \vert f(x) - g(x) \vert $")
        axs[1].set_xscale("log")
        axs[1].set_xlabel("x")
        axs[1].grid(True)

        plt.show()


if __name__ == "__main__":
    app.run(main)
