# Chapter7: Moving Beyond Linearity

## 7.1 Polynomial Regression and Step Functions

## Polynomial Regression


$$
y_i = \beta_0 + \beta_1x_i+\beta_2x_i^2+\beta_3x_i^3+\cdots+\beta_dx_i^d+\epsilon_i
$$


* Creat new variables $$X_1=X, X_2=X^2,$$ etc andthen treat as multiple linear regression.
* Not really interested in the coefficients; more interested in the fitted function values at an value $$x_0$$:

  $$
  \hat{f}(x_0)=\hat{\beta_0}+\hat{\beta_2}x_0^2+\hat{\beta_3}x_0^3+\hat{\beta_4}x_0^4
  $$

* Since $$\hat{f}(x_0)$$ is a linear function of the $$\hat{\beta_l}$$, can get a simple expression for _**pointwise-variances**_ $$\mathrm{Var}[\hat{f}(x_0)]$$ at any value $$x_0$$.
* We either fix the degree _**d**_ at some reasonably low value, else use cross-validation to choose _**d**_.
* Caveat: polynomials have notoeious tail behavior -- very bad for extrapolation.
* Can fit using &lt;s&gt;



