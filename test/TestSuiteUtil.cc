#include <gtest/gtest.h>
#include "util.h"

void CompareVectors(const std::vector<double>& a,
                    const std::vector<double>& b) {
  EXPECT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_DOUBLE_EQ(a[i], b[i]);
  }
}

TEST(UtilTest, ComputeGamma) {
  double h{0.1};
  std::vector<double> d{1, 2, 2.5};
  std::vector<double> result{0.9048374180359595, 0.8187307530779818,
                             0.7788007830714049};
  std::vector<double> gamma = ComputeGamma(d, h);
  CompareVectors(result, gamma);
}

TEST(UtilTest, ComputeBeta1) {
  double h{0.1};
  std::vector<double> d{1, 2, 2.5};
  std::vector<double> c{3, 1, 2};
  std::vector<double> gamma{5, 3, 1};
  std::vector<double> result{122.99999999999999, 5.5, 0.8};
  std::vector<double> beta1 = ComputeBeta1(d, c, gamma, h);
  CompareVectors(result, beta1);
}

TEST(UtilTest, ComputeBeta2) {
  double h{0.1};
  std::vector<double> d{1, 2, 2.5};
  std::vector<double> c{3, 1, 2};
  std::vector<double> gamma{5, 3, 1};
  std::vector<double> result{-135, -6.499999999999999, -0.8};
  std::vector<double> beta2 = ComputeBeta2(d, c, gamma, h);
  CompareVectors(result, beta2);
}

TEST(UtilTest, ComputeBeta) {
  double c_inf{1};
  std::vector<double> beta1{1, 2, 3};
  std::vector<double> beta2{4, 5, 6};
  double result{22};
  double beta = ComputeBeta(beta1, beta2, c_inf);
  EXPECT_DOUBLE_EQ(result, beta);
}
