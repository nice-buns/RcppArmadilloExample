// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil;
// -*-

// [[Rcpp::depends(RcppParallel)]]
#include "tbb/parallel_for.h"
#include <RcppParallel.h>
#include <oneapi/tbb/blocked_range.h>



// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include <RcppArmadillo.h>


// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//

//
// Found this code online!
//


// [[Rcpp::export]]
Rcpp::List fastLm(const arma::mat& X, const arma::colvec& y) {
    int n = X.n_rows, k = X.n_cols;
    arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
    arma::colvec res  = y - X*coef;           // residuals
    // std.errors of coefficients
    double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - k);
    arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
    return Rcpp::List::create(
      Rcpp::Named("coefficients") = coef,
      Rcpp::Named("stderr")       = std_err,
      Rcpp::Named("df.residual")  = n - k
    );
}


// [[Rcpp::export]]
void squareInParallel(Rcpp::NumericVector v) {

  auto vSize = v.size();
  double *basePtr = &(v[0]);
  Rcpp::Rcout << "Gave me a vector of size: " << vSize << "\n";

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vSize),
                    [basePtr](tbb::blocked_range<size_t> const &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        basePtr[i] = basePtr[i] * basePtr[i];
                      };
                      return;
                    });

  return;
}

// [[Rcpp::export]]
void squareInSerial(Rcpp::NumericVector v) {

  auto vSize = v.size();
  Rcpp::Rcout << "Gave me a vector of size: " << vSize << "\n";

for (auto i = 0; i < vSize; ++i){
    v[i] = v[i]*v[i];
}



  return;
}