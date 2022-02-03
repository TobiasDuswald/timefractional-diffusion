#ifndef UTIL_H_
#define UTIL_H_

#include <vector>

/// Computes the vector \gamma_k from the expansion coefficients d_k as in eq.
/// (5.13) in Khristenko & Wohlmuth (2022).
std::vector<double> ComputeGamma(const std::vector<double>& d, double h);

/// Computes the vector \beta^1_k from the expansion coefficients d_k, c_k and
/// \gamma_k as in eq. (5.13) in Khristenko & Wohlmuth (2022).
std::vector<double> ComputeBeta1(const std::vector<double>& d,
                                 const std::vector<double>& c,
                                 const std::vector<double>& gamma, double h);

/// Computes the vector \beta^2_k from the expansion coefficients d_k, c_k and
/// \gamma_k as in eq. (5.13) in Khristenko & Wohlmuth (2022).
std::vector<double> ComputeBeta2(const std::vector<double>& d,
                                 const std::vector<double>& c,
                                 const std::vector<double>& gamma, double h);

/// Computes the constant beta from the computed coefficients beta^1 and beta^2
/// and the expansion coefficient c_inf as in eq. (5.13) in Khristenko &
/// Wohlmuth (2022).
double ComputeBeta(const std::vector<double>& beta_1,
                   const std::vector<double>& beta_2, double c_inf);

#endif