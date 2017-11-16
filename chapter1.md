# Chapter4: Classification

## 4.1  Introduction to Classification

* Qualitative variables take values in an unordered set $$\mathcal{C}$$

* Given a feature vector $$X$$ and a qualitative response $$Y$$ taking values in the set $$\mathcal{C}$$, the classification task is to build a function $$C(X)$$ that takes as input the feature vector X and predicts its value for $$Y$$; i.e. $$C(X) \in \mathcal{C}$$

* Often we are more interested in estimating the _**probabilities**_ that $$X$$ belongs to each category in $$\mathcal{C}$$.

Suppose for the **Default** classification task that we code


$$
Y=\begin{cases} 0 \quad  \text{if  No} \\ 1 \quad \text{if  Yes} \end{cases}
$$


Can we simply perform a linear regression of $$Y$$ on $$X$$ and classify as **Yes **if $$\hat{Y}>0.5$$

* In this case of a binary outcome, linear regression does a good job as a classifier, and is equivalent to _**linear discriminant analysis**_ which we discuss later.

* Since in the population $$E(Y|X=x)=\mathrm{Pr}(Y=1|X=x)$$, we might think that regression is perfect for this task.

* However, _**linear**_ regression might produce probabilities less than zero or bigger than one. _**Logistic regression**_ is more appropriate.

## 4.2  Logistic Regression and Maximum Likelihood

### Logistic Regression

Logistic regression uses the form:


$$
p(X)=\frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}, \quad \text{where}\;\; p(X)=\mathrm{Pr}(Y=1|X=x)
$$


A bit of rearrangement gives:


$$
log\bigg(\frac{p(X)}{1-p(X)}\bigg)=\beta_0+\beta_1X
$$


This monotone transformation is called the _**log odds**_ or _**logit**_ transformation of$$p(X)$$

Logistic regression ensures that our estimate for $$p(X)$$ lies between 0 and 1.

### Maximum Likelihood

We use maximum likelihood to estimate the parameters:


$$
l(\beta_0,\beta_1)=\prod_{i:y_i=1}p(x_i)\prod_{i:y_i=0}(1-p(x_i))
$$


### Logistic regression with several variables


$$
log\bigg(\frac{p(X)}{1-p(X)}\bigg)=\beta_0+\beta_1X+\cdots+\beta_pX_p \\p(X)=\frac{e^{\beta_0+\beta_1X+\cdots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X+\cdots+\beta_pX_p}}
$$


## 4.3  Multivariate Logistic Regression

The symmetric form:


$$
\mathrm{Pr}(Y=k|X=x)=\frac{e^{\beta_{0k}+\beta_{1k}X+\cdots+\beta_{pk}X_p}}{\sum_{l=1}^Ke^{\beta_{0l}+\beta_{1l}X+\cdots+\beta_{pl}X_p}}
$$


Here there is a linear function for_** each**_ class

Multiclass logistic regression is also referred to as _**multinomial**_ regression.

## 4.4  Discriminat Analysis

Here the approach is to model the distribution of $$X$$ in each of the classes separately, and then use _**Bayes theorem**_ to flip things around and obtain $$\mathrm{Pr}(Y|X)$$.

When we use **normal \(Gaussian\) distributions** for each class, this leads to _**linear or quadratic discriminant analysis.**_

However, this approach is quite general, and other distributions can be used as well. We will focus on normal distributions.

### **Bayes theorem for classification**

Bayes theorem:


$$
\mathrm{Pr}(Y=k|X=x)=\frac{\mathrm{Pr}(X=x|Y=k)\mathrm{Pr}(Y=k)}{\mathrm{Pr}(X=x)}
$$


One writes this slightly differently for discriminant analysis:


$$
\mathrm{Pr}(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^{K}\pi_lf_l(x)}
$$


where,

* $$f_k(x)=\mathrm{Pr}(X=x|Y=k)$$   is the _**density**_ for $$X$$ in class $$k$$. Here we will use normal densities for these, separately in each class.

* $$\pi_k=\mathrm{Pr}(Y=k)$$is the _**marginal**_ or \_**prior **\_probability for class $$k$$.

### Why discriminant analysis?

* When the classes are _**well-separated**_, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.

* If $$n$$ is _**small **\_and the distribution of the predictors _$$X$$_ _**is approximately normal in each of the classes,**\_ the linear discriminant model is again more stable than the logistic regression model

* Linear discriminant analysis is popular when we have_** more than two response classes**_, because it also provides low-dimensional views of the data.

### Linear Discriminant Analysis when p = 1

The Gaussian density has the form:


$$
f_k(x)=\frac{1}{\sqrt{2\pi}\sigma_k}e^{-\frac{1}{2}\big(\frac{x-\mu_k}{\sigma_k}\big)^2}
$$


Here $$\mu_k$$ is the mean, and $$\sigma_k^2$$ the variance \(in class $$k$$\). We will assume that all the $$\sigma_k=\sigma$$ are the same.

Plugging this into Bayes formula, we get a rather complex expression for: $$p_k(x)=\mathrm{Pr}(Y=k|X=x)$$:


$$
p_k(x)=\frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\big(\frac{x-\mu_k}{\sigma}\big)^2}}{\sum_{l=1}^K\pi_k \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\big(\frac{x-\mu_l}{\sigma}\big)^2}}
$$


Happily, there are simplifications and cancellations.

To classify at the value $$X=x$$, we need to see which of the $$p_k(x)$$ is largest. Taking logs, and discarding terms that do not depend on $$k$$, we see that this is equivalent to assigning $$x$$ to the class with the _**largest discriminant score**_:


$$
\delta_k(x)=x\frac{\mu_k}{\sigma^2}-\frac{\mu_k^2}{2\sigma^2}+\text{log}(\pi_k)
$$


Note that $$\delta_k(x)$$ is a_** linear function**_ of $$x $$.

The solution of find the threshold of is  _**decision boundary**_  or _**bayea boundary**_

If there are $$K=2$$ classes and $$\pi_1=\pi_2=0.5$$, then one can see that the _**decision boundary**_ is at: $$x = \frac{\mu_1+\mu_2}{2}$$

### Estimating the parameters


$$
\begin{split} \hat{\pi}_k&=\frac{n_k}{n}\\ \hat{\mu_k}&=\frac{1}{n_k} \sum_{i:y_i=k}x_i\\ \hat{\sigma}^2&=\frac{1}{n-K}\sum_{k=1}^{K}\sum_{i:y_i=k}(x_i-\hat{\mu}_k)^2\\ &=\sum_{k=1}^{K} \frac{n_k-1}{n-K}\hat{\sigma}_k^2 \end{split}
$$


whrere $$\hat{\sigma}_k^2=\frac{1}{n_k-1}\sum_{i:y_i=k}(x_i-\hat{\mu})_k^2$$ is the usual formula for the estimated variance in the $$k$$th class.

### Linear Discriminant Analysis when p &gt; 1

Density:


$$
f(x)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}
$$


Discriminant function:


$$
\delta_k(x)=x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+\mathrm{log}\pi_k
$$


is a linear function

When there are $$K$$ classes, linear discriminant analysis can be viewed exactly in a $$K-1$$ dimensional plot.

### From$$\delta_k(x)$$ to probabilities

Once we have estimates $$\hat{\delta}_k(x)$$, we can turn these into estimates for class probabilities:


$$
\widehat{Pr}(Y=k|X=x)=\frac{e^{\hat{\delta}_k(x)}}{\sum_{l=1}^{K}e^{\hat{\delta}_l(x)}}
$$


So classifying to the largest $$\hat{\delta}_k(x)$$ amounts to classifying to the class for which $$\widehat{Pr}(Y=k|X=x)$$ is largest.

**False positive rate**: The fraction of negative examples that are classified as positive

**False negative rate**: The fraction of positive examples that are classified as negative

![](/assets/屏幕快照 2017-11-15 下午8.09.44.png)

always want false positive error low

### Other forms of Discriminant Analysis


$$
\mathrm{Pr}(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^{K}\pi_lf_l(x)}
$$


When $$f_k(x)$$ are Gaussian densities, with the same covariance matrix $$\Sigma$$ in each class, this leads to linear discriminant analysis. By altering the forms for $$f_k(x)$$, we get different classifiers.

* With Gaussians but different $$\Sigma_k$$ in each class, we get _**quadratic discriminant analysis.**_

* With $$f_k(x)=\prod_{j=1}^pf_{jk}(x_j)$$ \(conditional independence model\) in each class we get **naive Bayes**. For Gaussian this means the $$\Sigma_k$$ are diagonal.

* Many other forms, by proposing specific density models for $$f_k(x)$$, including nonparametric approaches.

$$\delta_k(x)=-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\text{log}\pi_k$$

#### naive Beyes:

Assumes features are independent in each class.

Useful when $$p$$ is large, and so multivariate methods like QDA and even LDA break down.

* Gaussian naive Bayes assumes each $$\Sigma_k$$ is diagonal:
  $$
  \delta_k(x) \propto \text{log}\bigg[\pi_k\prod_{j=1}^p f_{kj}(x_j)  \bigg]= -\frac{1}{2}\sum_{j=1}^{p}\frac{(x_j-\mu_{kj})^2}{\sigma_{kj}^2}+\text{log}\pi_k
  $$



