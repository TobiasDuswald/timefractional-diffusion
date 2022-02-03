#include "util.h"
#include <cassert>
#include <cmath>
#include <numeric>

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