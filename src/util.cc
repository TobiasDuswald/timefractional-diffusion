#include "util.h"
#include <cnpy.h>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

std::vector<double> ComputeGamma(const std::vector<double>& d, double h) {
  std::vector<double> gamma(d.size());
#pragma omp simd
  for (size_t k = 0; k < d.size(); k++) {
    gamma[k] = std::exp(-d[k] * h);
  }
  return gamma;
}

std::vector<double> ComputeBeta1(const std::vector<double>& d,
                                 const std::vector<double>& c,
                                 const std::vector<double>& gamma, double h) {
  assert(d.size() == c.size());
  assert(c.size() == gamma.size());
  std::vector<double> beta1(d.size());
#pragma omp simd
  for (size_t k = 0; k < d.size(); k++) {
    beta1[k] = gamma[k] - (1 - d[k] * h);
    beta1[k] *= c[k];
    beta1[k] /= d[k] * d[k] * h;
  }
  return beta1;
}

std::vector<double> ComputeBeta2(const std::vector<double>& d,
                                 const std::vector<double>& c,
                                 const std::vector<double>& gamma, double h) {
  assert(d.size() == c.size());
  assert(c.size() == gamma.size());
  std::vector<double> beta2(d.size());
#pragma omp simd
  for (size_t k = 0; k < d.size(); k++) {
    beta2[k] = 1 - (1 + d[k] * h) * gamma[k];
    beta2[k] *= c[k];
    beta2[k] /= d[k] * d[k] * h;
  }
  return beta2;
}

double ComputeBeta(const std::vector<double>& beta_1,
                   const std::vector<double>& beta_2, double c_inf) {
  assert(beta_1.size() == beta_2.size());
  double beta = std::accumulate(beta_1.begin(), beta_1.end(), c_inf);
  beta = std::accumulate(beta_2.begin(), beta_2.end(), beta);
  return beta;
}

std::vector<double> ReadNPYVector(std::string filename) {
  // Catch error if file does not exist.
  if (!std::filesystem::exists(filename)) {
    throw std::invalid_argument(
        "<ReadNPYVector> Cannot resolve path - file does not exist. " +
        filename);
  }
  // Load NPY array
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  // Throw exeption for anyting else than 1 dimensional NPY arrays
  if (arr.shape.size() != 1u) {
    throw std::invalid_argument(
        "<ReadNPYVector> Received array with wrong dimension.");
  }
  // Rearrange information to a std::vector
  std::vector<double> x(arr.shape[0]);
  double* x_ptr{arr.data<double>()};
  for (size_t i = 0; i < arr.shape[0]; i++) {
    x[i] = x_ptr[i];
  }
  return x;
}
