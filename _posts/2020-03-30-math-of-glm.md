---
layout: post
title:  "Mathematics of Generalized Linear Models"
date:   2020-03-30 19:10:30 -0500
permalink: "glm-math"
categories: statistics
tags: [glm, fitting, generalized linear model]
mathjax: true
---

_Last updated {{ page.last-modified-date | date: '%B %d, %Y' }}_

---
## Introduction

A generalized linear model (GLM) describes the expected value of a response
variable \\(y\\) with a link function \\(g\\) and a linear combination of \\(K\\) feature
variables \\(x_k\\).
\\[ g(E[y]) \sim \beta_0 + \beta_1 x_1 + \ldots + \beta_K x_K \\]
The definition \\(x_0 \equiv 1\\) is used so that with \\(\mathbf{x}^\intercal \equiv [1, x_1,
\ldots, x_K]\\) and \\(\boldsymbol{\beta}^\intercal \equiv [\beta_0, \beta_1, \ldots
\beta_K]\\) the above equation is written as the following.
\\[ g(E[y]) \sim \mathbf{x} \cdot \boldsymbol{\beta}\\]

For \\(N\\) observations the response variables can be expressed as a vector
\\(\mathbf{y}^\intercal \equiv [y^{(1)}, \ldots, y^{(N)}]\\) and the data matrix is
\begin{equation}
\mathbf{X} \equiv \begin{pmatrix}
1 & x_1^{(1)} & \ldots & x_K^{(1)} \\\\\
1 & x_1^{(2)} & \ldots & x_K^{(2)} \\\\\
\vdots & \vdots & \ddots & \vdots \\\\\
1 & x_1^{(N)} & \ldots & x_K^{(N)}
\end{pmatrix}
\end{equation}
so that the matrix components are \\(X_{ik} = x_{k}^{(i)}\\).

---
## Exponential family of distributions

The exponential family is a useful class that includes many common elementary
distributions. With a parameterization \\(\boldsymbol{\theta}\\) and sufficient
statistics \\(\mathbf{T}(y)\\), the density function of each can be written in a
canonical form with a function \\(\boldsymbol{\eta}(\boldsymbol{\theta})\\) where
\\(\boldsymbol{\eta}\\) are the _natural parameters_ of the distribution.
\\[ f(y; \boldsymbol{\theta}) = \exp\left[ \boldsymbol{\eta}(\boldsymbol{\theta})
\cdot \mathbf{T}(y) - A(\boldsymbol{\theta}) + B(y) \right] \\]

The function \\(A\\) is the log partition function and is particularly useful when
expressed in terms of the natural parameters.
\\[ A(\boldsymbol{\eta}) = \log \left[ \int dy \, e^{\boldsymbol{\eta} \cdot \mathbf{T}(y) + B(y)} \right] \\]
The expectation value of the sufficient statistics is given by the gradient of
\\(A\\) with respect to the natural parameters.
\\[ \textrm{E}[\mathbf{T}(y)] = \nabla A(\boldsymbol{\eta}) \\]
The covariance matrix is given by the Hessian of the log partition function.
\\[
\textrm{Cov}[\mathbf{T}(y)] = \nabla \nabla^\intercal A(\boldsymbol{\eta})
\\]

When the sufficient statistic \\(T\\) and the function \\(\eta\\) are both the identity,
the distribution is said to be in the _natural exponential family_.
\\[ f(y; \theta) = \exp[\theta y - A(\theta) + B(y)] \\]
If the canonical link function is used so that \\(\theta = \mathbf{x} \cdot
\boldsymbol{\beta}\\) then the link function is determined by the derivative of
the log-partition function.
\\[ g^{-1}(\mathbf{x}\cdot\boldsymbol{\beta}) = \textrm{E}[y|\mathbf{x}, \boldsymbol{\beta}] = \frac{\partial
A}{\partial \theta} \\]

---
## Likelihood function

For a dataset of \\(N\\) observations \\(\\{(y^{(i)}, \mathbf{x}^{(i)}) | i \in [1,
\ldots, N]\\}\\) the total log-likelihood function using a canonical-form
exponential distribution for \\(y\\) is
\\[l(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) = \sum_{i=1}^N
\boldsymbol{\eta}^{(i)}(\mathbf{x}^{(i)}\cdot\boldsymbol{\beta}) \cdot
\mathbf{T}(y^{(i)}) - A\left(\boldsymbol{\eta}^{(i)}(\mathbf{x}^{(i)} \cdot \boldsymbol{\beta}) \right)\\]
where the \\(B(y)\\) terms have been excluded as they do not affect the dependency
of the likelihood on \\(\boldsymbol{\beta}\\).

The gradient with respect to \\(\boldsymbol{\beta}\\) is
\begin{equation}
\nabla_\beta l = \sum_{i=1}^N \mathbf{x}^{(i)} \left( \mathbf{T}(y^{(i)}) -
\textrm{E}[\mathbf{T}(y^{(i)})|\boldsymbol{\eta}] \right) \cdot
\boldsymbol{\eta}'
\end{equation}
where the fact that \\(\nabla_\eta A(\boldsymbol{\eta}) = E[y]\\). With a canonical
link function, \\(\eta = \mathbf{x}\cdot\boldsymbol{\beta}\\) and \\(\eta' = 1\\) (is
this always true?).

The Hessian is given by
\begin{equation}
\nabla_\beta \nabla^\intercal_\beta l = \sum_{i=1}^N \mathbf{x}^{(i)}
\mathbf{x}^{\intercal(i)} \left[ - \boldsymbol{\eta}' \cdot
\textrm{Cov}[\mathbf{T}(y^{(i)})|\boldsymbol{\eta}] \cdot \boldsymbol{\eta}' +
\left( \mathbf{T}(y^{(i)}) - \nabla_\eta A(\boldsymbol{\eta}) \right) \cdot
\boldsymbol{\eta}'' \right]
\end{equation}
where it should be noted that the term in the brackets is a scalar for each
individual sample \\(i\\).

For many applications the natural exponential form can be used so the sufficient
statistic is simply a scalar value of the response variable \\(\mathbf{T}(y) = y\\)
and the canonical link function is used so that \\(\eta = \mathbf{x} \cdot
\boldsymbol{\beta}\\) and \\(g(\textrm{E}[y]) = \eta\\). The above equations simplify
and can be expressed easily in matrix form to include the sum over observations.
\begin{align}
\nabla_\beta l &= \mathbf{X}^\intercal \left[ \mathbf{y} - g^{-1}(\mathbf{X}\boldsymbol{\beta})\right] \\\\\
\nabla_\beta \nabla_\beta^\intercal l &= - \mathbf{X}^\intercal \mathbf{S} \mathbf{X} \\
\end{align}
In the above expressions, the inverse link function \\(g^{-1}\\) is applied
element-wise and the variance matrix is diagonal with \\(S_{ii} =
\textrm{Var}[y^{(i)}|\mathbf{x}^{(i)} \cdot \boldsymbol{\beta}]\\).

---
## Dispersion parameter

A useful generalization is to use an _overdispersed_ exponential family. The
so-called dispersion is parameterized by the variable \\(\phi\\).
\\[ \log f(y; \boldsymbol{\theta}, \phi) = \frac{\boldsymbol{\eta}(\boldsymbol{\theta})
\cdot \mathbf{T}(y) - A(\boldsymbol{\theta})}{\phi} + B(y, \phi) \\]
One useful feature of this formalism is that the dispersion parameter can
sometimes absorb on of the parameters of a multi-parameter distribution, such as
the variance in a normal distribution, allowing the regression to be agnostic of
the absorbed parameter.

The first and second moments of \\(\mathbf{T}(y)\\) become the following.
\begin{align}
\textrm{E}[\mathbf{T}(y)] &= \nabla A(\boldsymbol{\eta}) \\\\\
\textrm{Cov}[\mathbf{T}(y)] &= \phi \nabla \nabla^\intercal A(\boldsymbol{\eta})
\end{align}
Note that the expression for the expectation value has not changed, but the
variance picks up a factor of \\(\phi\\). In the natural case where
\\(\mathbf{T}(y) = y\\), the second derivative of \\(A\\) with respect to the
natural parameter \\(\eta\\) is called the _variance function_ \\(V(\mu)\\) when
it can be written as a function of the predicted mean of the response \\(\mu =
g^{-1}(\mathbf{x}\cdot\boldsymbol{\beta})\\).
\\[ V(\mu) = \left. \frac{\partial^2 A(\eta)}{\partial \eta^2} \right|_{\eta = \eta(g(\mu))} \\]
The variance function, like the dispersion parameter, is unique only up to a
constant due to the following relationship.
\\[ \textrm{Var}[y] = \phi V(\mu) \\]

Allowing the dispersion parameter to be different for each observation also
provides a consistent approach to give each observation separate weights. The
likelihood and its derivatives are adjusted by a factor of \\(1/\phi^{(i)}\\)
for each sample. With the weight matrix \\(\mathbf{W}\\) defined as a diagonal
matrix such that \\( W_{ii} = 1/\phi^{(i)} \\), the gradient and Hessian
of the likelihood becomes in the natural case
\begin{align}
\nabla_\beta l &= \mathbf{X}^\intercal \mathbf{W} \left[ \mathbf{y} -
g^{-1}(\mathbf{X}\boldsymbol{\beta}) \right] \\\\\
\nabla_\beta \nabla_\beta^\intercal l &= - \mathbf{X}^\intercal \mathbf{W} \mathbf{S} \mathbf{X}
\end{align}

Correlations between different observations \\(y^{(i)}\\) could be induced by
allowing off-diagonal terms in \\(\mathbf{W}\\). Note that \\(\mathbf{W}\\) is
reminiscent of the inverse covariance matrix, at least in the multivariate
normal distribution. Some additional work should be done to derive a consistent
approach for a general response function, but for now I note that the Hessian
still appears symmetric so long as \\(\mathbf{W}\\) is.

---
## Iteratively re-weighted least squares

Iteratively re-weighted least squares (IRLS) is a useful tool for fitting GLMs
because it is typically relatively straightforward to compute the Jacobian and
Hessian of the likelihood function. The step \\(\Delta \boldsymbol{\beta}\\) in
the space of parameters \\(\boldsymbol{\beta}\\) is given by the solution to
\\[ \mathbf{H}(\boldsymbol{\beta}) \cdot \Delta\boldsymbol{\beta} = -
\mathbf{J}(\boldsymbol{\beta}) \\]
where \\(\mathbf{J}\\) and \\(\mathbf{H}\\) are the Jacobian (gradient) and
Hessian of the log-likelihood function, respectively. The Hessian does not have
to be inverted completely; efficient linear algebra procedures exist to solve
symmetric matrix equations of this form.

This procedure is similar to Newton's method for the gradient, and (up to
potential numerical issues) it will move towards the point with zero gradient so
long as the likelihood is a concave function. Fortunately in GLM applications
this condition typically holds, especially if the canonical link function is
used.

In the case of a canonical link function with scalar sufficient statistic
\\(y\\), this update rule for \\(\Delta \boldsymbol{\beta}\\) becomes the
following.
\begin{equation}
\left( \mathbf{X}^\intercal \mathbf{S} \mathbf{X} \right) \Delta
\boldsymbol{\beta} = \mathbf{X}^\intercal \left[ \mathbf{y} -
g^{-1}(\mathbf{X}\boldsymbol{\beta})\right]
\end{equation}
When a weight matrix \\(\mathbf{W}\\) is included to include the observations
with different weights, a simple adjustment is needed.
\begin{equation}
\left( \mathbf{X}^\intercal \mathbf{W} \mathbf{S} \mathbf{X} \right) \Delta
\boldsymbol{\beta} = \mathbf{X}^\intercal \mathbf{W} \left[ \mathbf{y} -
g^{-1}(\mathbf{X}\boldsymbol{\beta})\right]
\end{equation}
A Hermitian-solve algorithm can be applied to generate successive steps for
\\(\boldsymbol{\beta}\\) until a desired tolerance is reached.

---
## TODO Case studies

### Ordinary least squares (Gaussian)

The probability distribution function (PDF) of a normally-distributed variable
with the standard parameterization variable is
\\[ f(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(y - \mu)^2}{2\sigma^2}} \\]
or equivalently:
\begin{align}
\log f(y; \mu, \sigma^2) &= -\frac{(y-\mu)^2}{2\sigma^2} - \frac{1}{2} \log (2\pi\sigma^2) \\\\\
&= \frac{\mu y - \frac{\mu^2}{2}}{\sigma^2} - \frac{y^2}{2\sigma^2} -
\frac{1}{2} \log (2\pi\sigma^2)
\end{align}
The common approach is to use a natural description that uses only the linear
term in \\(y\\), taking the dispersion parameter \\(\phi = \sigma^2\\) so that
\\(\eta = \mu\\), \\(A(\eta) = \frac{\eta^2}{2}\\), and \\(B(y, \phi) =
-\frac{y^2}{2\phi} - \log(2\pi\phi) \\).[^ols-mean-var]
The canonical link function is the identity because if \\(\eta =
\mathbf{x}^\intercal\boldsymbol{\beta}\\) we clearly have \\(\mu =
\mathbf{x}^\intercal \boldsymbol{\beta}\\).

Even with variable weights, the log-likelihood function is simple.
\\[ \log l = -\frac{1}{2} (\mathbf{y} - \mathbf{X}^\intercal
\boldsymbol{\beta})^\intercal \mathbf{W} (\mathbf{y} - \mathbf{X}^\intercal
\boldsymbol{\beta}) \\]
This form shows the equivalence to ordinary least squares (OLS).
The correlated case is easily included by letting \\(\mathbf{W}\\) be the
inverse covariance matrix of the observed \\(y^{(i)}\\). The
\\(\boldsymbol{\beta}\\) that maximizes the likelihood can be found
analytically,
\\[ \hat{\boldsymbol{\beta}} = (\mathbf{X}^\intercal \mathbf{W} \mathbf{X})^{-1}
\mathbf{X}^\intercal \mathbf{W} \mathbf{y} \\]
although if the estimated parameters are far from zero an additional IRLS step
may be useful for improved numerical accuracy.

An alternative application is to include both \\(y\\) and \\(y^2\\) in the
sufficient statistics \\(\mathbf{T}^\intercal(y) = [y, y^2]\\). This leads to
natural parameters \\(\boldsymbol{\eta}^\intercal = \left[ \frac{\mu}{\sigma^2},
-\frac{1}{2\sigma^2} \right] \\) with a trivial dispersion parameter \\(\phi =
1\\). Up to a constant the log-partition function is \\(A(\eta_1, \eta_2) =
-\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log (-\eta_2) \\). This is likely a good
application for a non-canonical link function for \\(\eta_2\\), for instance
\\(\eta_2 = - \exp(\mathbf{x}\cdot\boldsymbol{\beta}_2)\\), since \\(\sigma^2 >
0\\) so \\(\eta_2 < 0\\).

[^ols-mean-var]:
    Check that the mean and variance of \\(y\\) are \\(\mu\\) and
    \\(\sigma^2\\), respectively, by taking first and second derivatives of
    \\(A(\eta)\\) while making sure to include the dispersion parameter factor
    in the expression for the variance.

### Logistic (Bernoulli)

The PDF of the Bernoulli distribution is
\\[ f(y; p) = p^y (1-p)^{1-y} \\]
or equivalently
\begin{align}
\log f(y; p) &= y \log p + (1-y) \log(1-p) \\\\\
&= y \log\left(\frac{p}{1-p}\right) + \log(1-p)
\end{align}
where \\(y\\) takes a value of either 0 or 1.
The natural parameter is \\(\eta = \log\left(\frac{p}{1-p}\right) =
\textrm{logit}(p) \\) and as \\(p\\) is the mean of the Bernoulli distribution
the logit function is also the canonical link function.

TODO: complete

### Poisson

### Laplace (does this fit?)

### Exponential
\\[ f(y; \lambda) = \begin{cases}
\lambda e^{-\lambda y} & \textrm{for } y \geq 0 \\\\\
0 & \textrm{for } y < 0
\end{cases}\\]
The exponential distribution is a special case of the gamma distribution with
\\(\alpha = 1\\).

NOTE: This might be the best case to first investigate non-canonical link functions.

### Binomial with known \\(n\\)

### Gamma
The gamma distribution has two parameters, but it turns out that the shape
parameter \\(\alpha\\) is often treated as the same for every observation, allowing
for the \\(\beta\\) parameter to be predicted by
\\(\mathbf{x}\cdot\boldsymbol{\beta}\\).
[See these notes](https://civil.colorado.edu/~balajir/CVEN6833/lectures/GammaGLM-01.pdf).
This seems analogous to the situations in OLS where the variance \\(\sigma^2\\)
is the same for each data point so the minimization is unaffected by its value.

### Negative binomial

#### Known \\(r\\)

#### Unknown \\(r\\)

### Inverse Gaussian

---
## TODO Summary of GLMs for exponential familes

A summary table would be useful listing the response distribution, natural
parameters, partition function, canonical link function, variance function, etc.
for each common case.

---
## TODO Regularization

Consider the case where the response variable is linearly separable by the
predictor variables in logistic regression. The magnitude of the regression
parameters will increase without bound. This situation can be avoided by
penalizing large values of the parameters in the likelihood function. 

### L2 regularization

### L1 regularization

---
## TODO Goodness of fit
Compare log-likelihoods of fit model to saturated model

---
## TODO Numerical considerations

---
Footnotes:
