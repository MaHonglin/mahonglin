# Chapter7: Moving Beyond Linearity

## 7.1 Polynomial Regression and Step Functions

### Polynomial Regression


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

* Can fit using `y~poly(x,degree=3)`in formula.

### Step Functions


$$
C_1(X)=I(X<35), C_2(X)=I(35\leq X<50),\cdots,C_k=I(X\geq65)
$$


* Creates series of dummy variables representing each group.
* Useful way of creating interactions that are easy to interpret.For example, interation effect of **Year** and** Age**:

  $$
  I(\mathbf{Year}<2005)\cdot \mathbf{Age},\quad I(\mathbf{Year}\geq 2005)\cdot \mathbf{Age}
  $$


  would allow for different linear functions in each age category.

* In R:`I(year<2005)`or `cut(age,c(18,25,40,65,90))` .
* Choice of cutpoints or _**knots**_ can be problematic. For creating nonlinearities, smoother alternatives such as _**splines**_ are available.

## 7.2 Piecewise Polynomials and Splines

### Piecewise Polynomials

* Instead of a single polynomial in $$X$$ over its whole domain, we can rather use different polynomials in regions defined by knots.


  $$
  y_i = \begin{cases} \beta_{01}+\beta_{11}x_i+\beta_{21}x_i^2+\beta_{31}x_i^3+\epsilon_i, \quad x_i < c \\ \beta_{02}+\beta_{12}x_i+\beta_{22}x_i^2+\beta_{32}x_i^3+\epsilon_i, \quad x_i \geq c \end{cases}
  $$

* Better to add constraints to the polynomials, for continuity

### Linear Splines

_**A linear spline with knots at **_$$\xi_{k},\; k=1,\cdots,K$$_** is a piecewise linear polynomial continuous at each knot.**_

We can represent this model as:


$$
y_i = \beta_0 + \beta_1b_1(x_i)+\beta_2b_2(x_i)+\beta_3b_3(x_i)+\cdots+\beta_{K+1}b_{K+1}(x_i)+\epsilon_i
$$


where the $$b_k$$ are _**basis functions**_:


$$
\begin{split} b_1(x_i) &= x_i \\ b_{k+1}(x_i)&=(x_i - \xi_k)_+, \quad k=1,2,\cdots,K \end{split}
$$


### Cubic Splines

_**A cubic spline with knots at **_$$\xi_{k},\; k=1,\cdots,K$$_** is a piecewise cubic polynomial with continuous derivatives up to order 2 at each knot.**_

Again we can represent this model with truncated power basis functions:


$$
y_i = \beta_0 + \beta_1b_1(x_i)+\beta_2b_2(x_i)+\beta_3b_3(x_i)+\cdots+\beta_{K+3}b_{K+3}(x_i)+\epsilon_i
$$



$$
\begin{split} b_1(x_i) &= x_i \\b_2(x_i)&=x_i^2\\b_3(x_i)&=x_i^3\\b_4(x_i)&=(x_i-\xi_1)_+^3 \\b_{k+3}(x_i)&=(x_i - \xi_k)_+^3, \quad k=1,2,\cdots,K \end{split}
$$


### Natural Cubic Splines

A natural cubic spline extrapolates linearly beyond the boundary knots. This adds 4 = 2 × 2 extra constraints\(4 less basis\), and allows us to put more internal knots for the same degrees of freedom as a regular cubic spline.

Basis:


$$
\begin{split} b_1(x_i) &= x_i \\b_2(x_i)&=x_i^2\\b_3(x_i)&=d_1(x_i)-d_{K-1}(x_i)\\b_{k-2}(x_i)&=d_k(x_i)-d_{K-1}(x_i), \quad k=1,2,\cdots,K \end{split} \\ d_k(x)=\frac{(x_i-\xi_k)_+^3-(x_i-\xi_K)_+^3}{\xi_K-\xi_k}, \quad k=1,2,\cdots,K-1
$$


### Knot Placement

* A cubic spline with K knots has K + 4 parameters\(number of $$\beta_i$$\) or _**degrees of freedom.**_

* A natural spline with K knots has K  _**degrees of freedom.**_

* One strategy is to decide K, the number of knots, and then place them at appropriate quantiles of the observed $$X$$ , in R, use `bs(x, ...)` or `ns(age, df=14)` in package `splines`.

### Smooth Splines



