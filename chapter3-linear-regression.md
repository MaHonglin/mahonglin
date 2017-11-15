# Chapter3: Linear Regression

## 3.1  Simple Linear Regression and Confidence Intervals

* Linear regression is a simple approach to supervised learning. It assumes that the dependence of $$Y$$ on $${X_1}, \ldots ,{X_p}$$ is linear.

* We assume a modle:
  $$
  Y=\beta_0+\beta_1X+ \epsilon
  $$
  where $$\beta_0$$ and $$\beta_1$$ are two unknown constants that represent the _**intercept**_ and _**slope**_, also known as _**coefficients**_ or _**parameters**_, and $$\epsilon$$  is the error term.

* Given some estimates $$\hat{\beta_0}$$ and $$\hat{\beta_1}$$ for the model coefficients, we predict future sales using：
  $$
  \hat{y}=\hat{\beta}_0+\hat{\beta}_1x
  $$
  where $$\hat{y}$$ indicates a prediction of $$Y$$ on the basis of $$X=x$$. The _**hat **_symbol denotes an estimated value.

* Let $$\hat{y}_i=\hat{\beta}_0+\hat{\beta}_1x_i$$be the prediction for $$Y$$ based on the $$i$$th value of $$X$$. Then $$e_i = y_i - \hat{y}_i$$ represents the $$i$$th _**residual**_

   

* We define the residual sumof squares \(RSS\) as
  $$
  \mathrm{RSS}=e_1^2+e_2^2+\cdots+e_n^2
  $$
  or equivalently as:
  $$
  RSS=(y_1-\hat{\beta}_0-\hat{\beta}_1x_1)^2+(y_2-\hat{\beta}_0-\hat{\beta}_1x_2)^2+\cdots+(y_n-\hat{\beta}_0-\hat{\beta}_1x_n)^2
  $$
* The least squares approach chooses $$x = y$$ and $$\beta_1$$to minimize the RSS. The minimizing values can be shown to be
  $$
  \begin{split} \hat{\beta}_1&=\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n(x_i-\bar{x})^2} \\ \hat{\beta}_0&=\bar{y}-\hat{\beta}_1\bar{x} \end{split}
  $$
  where $$\bar{y}=\frac{1}{n}\sum_{i=1}^n y_i$$ and $$\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i$$ are the sample means.

* The standard error of an estimator reflects how it varies under repeated sampling. We have
  $$
  \mathrm{SE}(\hat{\beta}_1)^2=\frac{\sigma^2}{\sum_{i=1}^n(x_i-\bar{x})^2} , \quad \mathrm{SE}(\hat{\beta}_0)^2=\sigma^2\Big[ \frac{1}{n}+\frac{\bar{x}^2}{\sum_{i=1}^n(x_i-\bar{x})^2} \Big]
  $$
  where $$\sigma^2=\mathrm{Var}(\epsilon)$$

* These standard errors can be used to compute _**confidence intervals**_. A 95% confidence interval is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameter. It has the form
  $$
  \hat{\beta}\pm\mathrm{SE}(\hat{\beta})
  $$

## 3.2  Hypothesis Testing

* Standard errors can also be used to perform _**hypothesis tests**_ on the coefficients. The most common hypothesis test involves testing the _**null hypothesis**_ of
* * $$H_0$$: There is no relationship between $$X$$ and $$Y$$ versus the _**alternative hypothesis**_
  * $$H_1$$:  There is some relationship between $$X$$ and $$Y$$

* Mathematically, this corresponds to testing:
  $$
  H_0 : \beta_1=0
  $$
  versus
  $$
  H_A : \beta_1 \ne 0
  $$
  since if $$\beta_1=0$$ then the model reduces to $$Y=\beta_0+\epsilon$$, and $$X$$ is not associated with $$Y$$

* To test the null hypothesis, we compute a _**t-statistic**_, given by
  $$
  t=\frac{\hat{\beta}_1-0}{\mathrm{SE}(\hat{\beta}_1)}
  $$

* This will have a t-distribution with n − 2 degrees of freedom, assuming $$\beta_1=0$$.

* Using statistical software, it is easy to compute the probability of observing any value equal to $$\lvert t \rvert$$or larger. We call this probability the _**p-value**_.

### Assessing the Overall Accuracy of the Model

* We compute the _**Residual Standard Error**_
  $$
  \mathrm{RSE}=\sqrt{\frac{1}{n-2}\mathrm{RSS}}=\sqrt{\frac{1}{n-2}\sum_{i=1}^n(y_i-\hat{y}_i)^2}
  $$

* _**R-squared**_ or fraction of variance explained is
  $$
  R^2=\frac{\mathrm{TSS}-\mathrm{RSS}}{\mathrm{TSS}}=1-\frac{\mathrm{RSS}}{\mathrm{TSS}}
  $$
  where $$\mathrm{TSS}=\sum_{i=1}^n(y_i-\hat{y})^2$$ is the _**total sum of squares.**_

* It can be shown that in this simple linear regression setting that $$R^2=r^2$$ , where r is the correlation between $$X$$ and $$Y$$ :
  $$
  r=\frac{\sum{i=1}^n(x_i-\hat{x})(y_i-\hat{y})}{\sqrt{\sum{i=1}^n(x_i-\hat{x})^2}\sqrt{\sum{i=1}^n(y_i-\hat{y})^2}}
  $$

## 3.3  Multiple Linear Regression and Interpreting Regression Coefficients

* Here our model is：
  $$
  Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\epsilon
  $$

* We interpret $$\beta_j$$ as the _**average**_ effect on $$Y$$ of a one unit increase in $$X_j$$, holding all _**other predictors fixed.**_

In hypothesis testing, use F-statistic:
$$
F=\frac{(\mathrm{TSS}-\mathrm{RSS})/p}{\mathrm{RSS}/(n-p-1)}\sim F{p,n-p-1}
$$


## 3.4  Model Selection and Qualitative Predictors

The most direct approach is called _**all subsets**_ or _**best subsets**_ regression: we compute the least squares fit for all possible subsets and then choose between them based on some criterion that balances training error with model size

We need an automated approach that searches through a subset of them. We discuss two commonly use approaches next.

### Forward selection

* Begin with the _**null model**_ — a model that contains an intercept but no predictors.

* Fit $$p$$ simple linear regressions and add to the null model the variable that results in the lowest **RSS**.

* Add to that model the variable that results in the lowest **RSS** amongst all two-variable models.

* Continue until some stopping rule is satisfied, for example when all remaining variables have a **p-value** above some **threshold**.

### Backward selection

* Start with all variables in the model.

* Remove the variable with the largest p-value — that is, the variable that is the least statistically significant.

* The new $$(p-1)$$-variable model is fit, and the variable with the largest p-value is removed.

* Continue until a stopping rule is reached. For instance, we may stop when all remaining variables have a significant **p-value** defined by some **significance threshold.**

## Qualitative Predictors

* Some predictors are not_** quantitative**_ but are _**qualitative**_, taking a discrete set of values. 
* These are also called _**categorical**_ predictors or _**factor variables**_

* There will always be **one fewer** _**dummy variable**_ than the number of _**levels**_. The level with no dummy variable — is known as the** **_**baseline**_.

## Extensions of the Linear Model

### Interactions

* The _**hierarchy principle**_：

* * **If we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.**

### N**onlinearity**

### Other Problems

Outliers 

Non-constant variance of error terms 

High leverage points 

Collinearity

