# Chapter 2: Statistical Learning

## 2.1 

Assume:
$$
Y=f(x)+\epsilon
$$
An ideal $$f(x)=E(Y\vert X)$$ is called _**regression function.**_

* Is the _**ideal**_ or _**optimal**_ predictor of $$Y$$ with regard to mean-squared prediction error:$$f(x)=E(Y\vert X)$$ is the function that minmizes $$E[(Y-g(X)^2)\vert X=x]$$ over all functions $$g$$ at all points $$X=x$$.
* $$\epsilon = Y-f(x)$$ is the _**irrducible**_ error 
* For many estimate $$\hat{f}(x)$$ of $$f(x)$$ , we have:
  $$
  E[(Y-\hat{f}(X)^2)\vert X=x]=[f(X)-\hat{f}(X)]^2+\mathrm{Var}(\epsilon)
  $$

To estimate$$f$$ , if we let:
$$
\hat{f}(x)=\mathrm{Ave}(Y\vert X \in \mathcal{N}(x))
$$
where $$\mathcal{N}(x)$$ is some _**neighborhood**_ of $$x$$.

* Nearest neighbor averaging can be pretty good for small p and large N.

* Nearest neighbor methods can be _**lousy**_ when p is large. Reason: the _**curse of dimensionality**_. Nearest neighbors tend to be far away in high dimensions.

* The _**linear**_ model is an important example of a parametric model:
  $$
  f_{L}(X)=\beta_0+\beta_1X_1+\beta_2X_2+,\cdots,+\beta_pX_p
  $$

* A linear model is specified in terms of p + 1 parameters 

* We estimate the parameters by fitting the model to training data.

* Although it is _**almost never correct**_, a linear model often serves as a good and interpretable approximation to the unknown true function $$f(X)$$

  .

## 2.2  Assessing Model Accuracy

Suppose we fit a model $$\hat{f}(x)$$ to some training data 

$$\mathrm{Tr}=\{x_i,y_i\}_1^N$$, and we wish to see how well it performs. 

* We could compute the average squared prediction error over Tr: 
  $$
  \mathrm{MSE_{Tr}}=\mathrm{Ave}_{i \in \mathrm{Tr}}[y_i-\hat{f}(x_i)]^2
  $$

This may be biased toward more overfit models. 

* Instead we should, if possible, compute it using fresh _**test**_ data$$x = y\mathrm{Te}=\{x_i,y_i\}_1^M$$: 
  $$
  \mathrm{MSE_{Te}}=\mathrm{Ave}_{i \in \mathrm{Te}}[y_i-\hat{f}(x_i)]^2
  $$

## 2.3  Bias-Variance Trade-off

Suppose we have fit a model $$\hat{f}(x)$$ to some training data _**Tr**_, and let $$(x_0,y_0)$$ be a test observation drawn from the population. If the true model is $$Y=f(X)+\epsilon (f(x)=E(Y\vert X=x))$$, then:

$$E\big(y_0-\hat{f}(x_0)\big)^2= \mathrm{Var}(\hat{f}(x_0))+[\mathrm{Bias}(\hat{f}(x_0))]^2+\mathrm{Var}(\epsilon)$$

The expectation averages over the variability of y0 as well as the variability in _**Tr**_. Note that $$\mathrm{Bias}(\hat{f}(x_0))=E[\hat{f}(x_0)]-f(x_0)$$.

Typically as the_** flexibility **_of $$\hat{f}$$ increases, its variance increases, and its bias decreases. So choosing the flexibility based on average test error amounts to a _**bias-variance trade-off**_.

