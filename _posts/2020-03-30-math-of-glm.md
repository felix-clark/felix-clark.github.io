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
\\[ g(E[y]) = \mathbf{x}^\intercal \boldsymbol{\beta}\\]
The _linear predictor_ is defined as \\(\omega = \mathbf{x}^\intercal
\boldsymbol{\beta} \\).

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
\boldsymbol{\eta}(\omega^{(i)}) \cdot
\mathbf{T}(y^{(i)}) - A\left(\boldsymbol{\eta}(\omega^{(i)}) \right)\\]
where the \\(B(y)\\) terms have been excluded as they do not affect the dependency
of the likelihood on \\(\boldsymbol{\beta}\\).

When multiple sufficient statistics are used \\(\mathbf{T}^\intercal(y) = [T^1
(y), T^2(y), \ldots]\\) the natural approach is to make each
natural parameter a function of a separate set of regression parameters
\\(\eta^a = \eta^a(\mathbf{x}^\intercal \boldsymbol{\beta}^a)\\).
The gradient with respect to \\(\boldsymbol{\beta}^a\\) is
\begin{equation}
\nabla_{\beta^a} l = \sum_{i=1}^N \mathbf{x}^{(i)} \left( T^a (y^{(i)}) -
\textrm{E}[T^a (y^{(i)})|\boldsymbol{\eta}] \right) \eta^{a\prime}
\end{equation}
where the fact that \\(\nabla_\eta A(\boldsymbol{\eta}) = E[\mathbf{T}(y)]\\) is used.
By definition, the canonical link function is the one that results from using
\\(\eta = \mathbf{x}\cdot\boldsymbol{\beta}\\) which results in \\(\eta' = 1\\).

The Hessian is given by
\begin{equation}
\nabla_{\beta^a} \nabla^\intercal_{\beta^b} l = \sum_{i=1}^N \mathbf{x}^{(i)}
\mathbf{x}^{\intercal(i)} \left\\{ - \eta^{a \prime}
\textrm{Cov}\left[T^{ab}(y^{(i)}) \middle| \boldsymbol{\eta}\right] \eta^{b \prime} +
\delta_{ab} \eta^{a\prime\prime}\left( T^b (y^{(i)}) -
\textrm{E}\left[T^b(y^{(i)}) \middle| \boldsymbol{\eta} \right] \right) \right\\}
\end{equation}
where \\(\delta_{ab}\\) is the Kronecker delta and there is no implicit
summation over indices \\(a, b\\) in the term in the curly brackets.

Often the sufficient statistic is just the response variable itself
\\(\mathbf{T}(y) = y\\). This allows for some simplification of the above
equations for the log-likelihood and its derivatives, while still allowing for a
general link function.
\begin{align}
l(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) &= \sum_{i=1}^N
\eta(\mathbf{x}^{\intercal(i)} \boldsymbol{\beta}) y^{(i)} -
A\left(\eta(\mathbf{x}^{\intercal(i)} \boldsymbol{\beta}) \right) \\\\\
\nabla_{\beta} l &= \sum_{i=1}^N \mathbf{x}^{(i)} \eta^{\prime} \left( y^{(i)} -
\textrm{E}[y^{(i)}|\eta] \right)\\\\\
\nabla_{\beta} \nabla^\intercal_{\beta} l &= \sum_{i=1}^N \mathbf{x}^{(i)}
\mathbf{x}^{\intercal(i)} \left\\{ - (\eta^{\prime})^2 \textrm{Var}\left[y^{(i)}
\middle| \eta\right] + \eta^{\prime\prime}\left( y^{(i)} -
\textrm{E}\left[y^{(i)} \middle| \eta \right] \right) \right\\}
\end{align}

If in addition the canonical link function is used, \\(\eta = \mathbf{x} \cdot
\boldsymbol{\beta}\\) and \\(g(\textrm{E}[y]) = \eta\\). The above equations
simplify further and can be expressed easily in matrix form to include the sum
over observations,
\begin{align}
l &= \mathbf{y}^\intercal \mathbf{X} \boldsymbol{\beta} - \sum_{i=1}^N
A\left(\omega^{(i)} \right) \\\\\
\nabla_\beta l &= \mathbf{X}^\intercal \left[ \mathbf{y} - g^{-1}(\mathbf{X}\boldsymbol{\beta})\right] \\\\\
\nabla_\beta \nabla_\beta^\intercal l &= - \mathbf{X}^\intercal \mathbf{S} \mathbf{X} \\
\end{align}
where the inverse link function \\(g^{-1}\\) is applied element-wise and the
variance matrix is diagonal with \\(S_{ii} = \textrm{Var}[y^{(i)}|\eta]\\).

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
Expressing the variance as a function of the mean is useful because it is immune
to changing the link function.
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
## Comments on computing the link and variance functions

In the standard description the link function generally relates the expected
value of the response variable to the linear predictor.
\\[ g(\textrm{E}[y|\omega]) = \omega \\]
Once the link function is known, its inverse can be used to quickly get the
expected value given the linear predictor.
\\[ \textrm{E}[y|\omega] = g^{-1}(\omega) \\]
The expected response value, or mean \\(\mu\\), is known either from knowledge
of the standard parameters \\(\boldsymbol{\theta}\\) or, assuming a single
sufficient statistic \\(y\\), through the first derivative of the log-partition
function as a function of the natural parameter \\(\eta\\).
\\[ \mu = \frac{\partial A(\eta)}{\partial \eta} \; \textrm{for } \mathbf{T}(y)
= [y]\\]
This expression can be inverted to get the natural parameter as a function of the
mean \\(\eta(\mu) = (A')^{-1}(\mu)\\).
To use the canonical link function, the natural parameter is equal to the linear
predictor so that \\( \eta_0(\omega) = \omega \\).
A different parameterization can be used for \\(\eta\\), which implies a change
in the link function.
Let the natural parameter be defined through an invertible transformation
\\(s\\) on the linear predictor.
\\[ \eta(\omega) = s(\omega) \\]
Given this transformation and the expression for \\(\eta\\) in terms of
\\(\mu\\), the effect of this transformation on the link function is given by
the following equation.
\\[ g(\mu) = s^{-1}\left( \eta(\mu) \right) \\]
The same function \\(\eta(\mu)\\) can be used to compute the variance function
\\(V(\mu)\\).
\\[ V(\mu) = \left. \frac{\partial^2 A(\eta)}{\partial \eta^2} \right|_{\eta = \eta(\mu)} \\]
Note that it is unaffected by any transformation \\(s\\).
In fact, by the [inverse function
theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem), if
\\(\eta(\mu) = t(\mu)\\), then the variance function is the reciprocal of the
derivative of \\(t\\).
\\[ V(\mu) = \frac{1}{\frac{\partial t}{\partial \mu}} \\]

If there are multiple sufficient statistics, these statements should all be
generalizable in the obvious way.
Instead of a parameter \\(\mu\\) we have a vector of expectation values, and the
generalized link function maps these multiple values to the set of linear
predictors, which have the same cardinality.
The multiple natural parameters can still be expressed as functions of these
expected values.
The variance function is now a symmetric matrix but there typically should be
nothing stopping it from being expressible in terms of the expected values of
the sufficient statistics; indeed, the multivariate version of the inverse
function theorem should still apply.

See [this
post](https://stats.stackexchange.com/questions/40876/what-is-the-difference-between-a-link-function-and-a-canonical-link-function)
for a helpful summary of non-canonical link functions, although be wary of the
differing notation.
Another note: When the canonical link function is used, \\(\mathbf{X}^\intercal
\mathbf{y}\\) is a sufficient statistic for the whole data set.

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
## Case studies

### Linear (Gaussian)

The probability distribution function (PDF) of a normally-distributed variable
with the standard parameterization is
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
The variance function is \\(V(\mu) = 1\\).

Even with variable weights, the log-likelihood function is simple.
\\[ l = -\frac{1}{2} (\mathbf{y} - \mathbf{X}^\intercal
\boldsymbol{\beta})^\intercal \mathbf{W} (\mathbf{y} - \mathbf{X}^\intercal
\boldsymbol{\beta}) \\]
This form shows the equivalence to weighted least squares (WLS), or ordinary
least squares (OLS) when the weights are identical for each point.
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
0\\) so \\(\eta_2 < 0\\). Here the variance function \\(V(\mu)\\) could possibly
be replaced by a covariance function of the two-component expected values of
\\(\mu\\) and \\(\sigma^2\\), but checking this should require some more careful
analysis. 

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
\textrm{logit}(p) \\) and as \\(p = \mu\\) is the mean of the Bernoulli
distribution the logit function is also the canonical link function.
In terms of the natural parameter the log-partition function is \\(A(\eta) =
\log\left(1 + e^\eta \right)\\).
With the simple convention of \\(\phi=1\\), the variance function is \\(V(\mu) =
\mu(1-\mu)\\).


### Poisson

The Poisson distribution for non-negative integer \\(y\\) is
\\[ f(y; \lambda) = \frac{\lambda^y e^{-\lambda}}{y!} \\]
or equivalently
\\[ \log f(y; \lambda) = y \log \lambda - \lambda - \log(y!) \\]
The mean and variance are both equal to \\(\lambda\\), so the natural parameter
is \\(\eta = \log \lambda\\) and the canonical link function is \\(g(\mu) = \log
\mu \\).
The log-partition is \\(A(\eta) = e^\eta \\) and \\(B(y) = -\log(y!)\\).
The dispersion parameter is taken to be \\(\phi = 1\\) and the variance function
is \\(V(\mu) = \mu\\).


### Exponential

While exponentially-distributed error terms are not common, this distribution is
still a useful case study because the distribution itself is mathematically
simple but the constraint on the distribution's parameter motivates a
non-canonical link function.
The PDF for an exponentially-distributed positive real value \\(y\\) is
\\[ f(y; \lambda) = \begin{cases}
\lambda e^{-\lambda y} & \textrm{for } y \geq 0 \\\\\
0 & \textrm{for } y < 0
\end{cases}\\]
with \\(\lambda > 0\\) and the log-PDF is
\\[ \log f(y; \lambda) = -\lambda y + \log \lambda \\]
where \\(y \geq 0\\). The natural parameter is \\(\eta = -\lambda\\) and the
log-partition function is \\(A(\eta) = - \log(-\eta)\\).
Since the mean of the distribution is \\(\mu = -\frac{1}{\eta} =
\frac{1}{\lambda}\\), the canonical link function is the negative inverse
\\(g_0(\mu) = -\frac{1}{\mu}\\).
The variance is \\(\textrm{Var}(y|\eta) = \frac{1}{\eta^2} \\) or
\\(\textrm{Var}(y|\lambda) = \frac{1}{\lambda^2} \\) so with the simple choice
of dispersion parameter \\(\phi = 1\\) the variance function is \\(V(\mu) =
\mu^2\\).

The canonical link function \\(-\frac{1}{\mu} = \mathbf{x}^\intercal
\boldsymbol{\beta}\\) does not prohibit values of \\(\mu < 0\\). The extent to
which this is problematic will depend on the data points themselves, but we will
show how to work around this as an example.

Consider the parameterization \\(\eta = -\exp(-\omega)\\) where \\(
\omega = \mathbf{x}^\intercal \boldsymbol{\beta}\\).
The mean and variance are \\(e^{\omega}\\) and \\(e^{2\omega}\\), respectively,
so the implied link function is logarithmic \\(g(\mu) = \log \mu =
\mathbf{x}^\intercal \boldsymbol{\beta} \\).
It is instructive to write out the likelihood and its first and second
derivatives in this parameterization[^exp-log-link-like].
\begin{align}
l &= \sum_{i=1}^N \left[ - \exp\left(-\omega^{(i)}\right) y^{(i)} - \omega^{(i)} \right]\\\\\
\nabla_{\beta} l &= \sum_{i=1}^N \mathbf{x}^{(i)} \left[ \exp\left(-\omega^{(i)}\right) y^{(i)} - 1 \right] \\\\\
\nabla_{\beta} \nabla^\intercal_{\beta} l &= - \sum_{i=1}^N \mathbf{x}^{(i)}
\mathbf{x}^{\intercal(i)} \exp\left(-\omega^{(i)}\right) y^{(i)}
\end{align}

[^exp-log-link-like]:
    Verify that the expression holds both by direct derivation and by applying
    the equations for the derivatives as a function of \\(\eta\\) and its derivatives.

Note that the exponential distribution is a special case of the gamma
distribution with \\(\alpha = 1\\). In practice using a gamma GLM may often be a
better choice, but a similar issue arises with negative values so non-canonical
link functions are often used there as well.


### Binomial (fixed \\(n\\))

The binomial distribution describes a variable \\(y\\) that can take values in a
finite range from \\(0\\) to \\(n\\) inclusive. It is in the exponential family
for a fixed \\(n\\), which is usually an obvious constraint based on the maximum
logically possible value for \\(y\\). The PDF is
\\[ f(y; n, p) =
\binom{n}{y} p^y (1-p)^{n-y}
\\]
or
\begin{align}
\log f(y; n, p) &= y \log p + (n-y) \log(1-p) + \log n! - \log y! - \log[(n-y)!] \\\\\
&= y \log \left( \frac{p}{1-p} \right) + n \log(1-p) + B(y, n)
\end{align}
and up to factors of \\(n\\) the analysis is similar to the logistic (Bernoulli)
case.
The natural parameter is \\(\eta = \textrm{logit}(p)\\), the log-partition
function is \\(A(\eta) = n \log(1 + e^\eta)\\), the mean is \\(\mu = np\\), and
the variance is \\(np(1-p)\\). With dispersion parameter \\(\phi = 1\\), the
variance function is \\(V(\mu) = \mu\left(1 - \frac{\mu}{n}\right)\\).
By writing \\(\eta\\) in terms of \\(\mu\\) it is seen that the canonical link
function is \\(g(\mu) = \log\left( \frac{\mu}{n - \mu}\right) \\).

### Gamma
The gamma distribution has two parameters, but it turns out that the shape
parameter \\(\alpha\\) is often treated as the same for every observation, allowing
for the \\(\beta\\) parameter to be predicted by
\\(\mathbf{x}\cdot\boldsymbol{\beta}\\).
[See these notes](https://civil.colorado.edu/~balajir/CVEN6833/lectures/GammaGLM-01.pdf).
This seems analogous to the situations in OLS where the variance \\(\sigma^2\\)
is the same for each data point so the minimization is unaffected by its value.

The PDF of the gamma distribution with the \\(\alpha, \beta\\) parameterization
is
\\[ f(y; \alpha, \beta) = \frac{\beta^\alpha y^{\alpha-1} e^{-\beta y}}{\Gamma(\alpha)} \\]
or equivalently
\\[
\log f(y; \alpha, \beta) = -\beta y + (\alpha-1) \log y + \alpha \log \beta -
\log \Gamma(\alpha)
\\]
where it is clear that the sufficient statistics should include at least one of
\\([y, \log y]\\) with corresponding natural parameters \\([-\beta,
\alpha-1]\\).

We will start the analysis a bit differently than the other distributions we've
looked at so far, in order to try to get to the simplest version as quickly as
possible.
The mean and variance of the gamma distribution are known to be
\\(\alpha/\beta\\) and \\(\alpha/\beta^2\\), respectively.
The variance could thus be written as either[^gamma-var] \\(\textrm{Var}[y] =
\frac{1}{\beta}\mu = \frac{1}{\alpha}\mu^2\\) with dispersion factor \\(\phi\\)
being the reciprocal of either \\(\beta\\) or \\(\alpha\\), respectively.
It turns out that the second choice, with \\(\phi = \frac{1}{\alpha}\\) and
\\(V(\mu) = \mu^2\\), is the most straightforward.
\\[ \log f(y; \mu, \phi) = \frac{-\frac{1}{\mu} y - \log \mu}{\phi} + B(y, \phi)
\\]
In fact, other than the \\(B(y, \phi)\\) term this looks just like the case of
the exponential distribution with \\(\lambda = 1/\mu\\), but with a non-unity
dispersion parameter that does not affect the regression itself.

[^gamma-var]:
    Actually by this logic the variance function could be any power of \\(\mu\\)
    with the right powers of \\(\alpha\\) and \\(\beta\\) as the dispersion
    factor, but for simplicity we'll focus on the options that allow \\(\phi\\)
    to be written only as a function of one or the other.
    
### Negative binomial (fixed \\(r\\))

The negative binomial distribution is an over-dispersed Poisson distribution, in
that its support is over the non-negative integers but its variance is greater
than its mean.
\begin{align}
f(y; r, p) &= \frac{p^r (1-p)^y}{ y B(r, y)} \\\\\
\log f(y; r, p) &= y \log(1-p) + r \log p - \log y - \log \textrm{Beta}(r, y)
\end{align}

Like the binomial distribution, the negative binomial distribution is in the
exponential family if and only if one of its standard parameters (\\(r\\)) is
held fixed. However, while the choice of \\(n\\) in a binomial distribution is
typically obvious, it is not usually so clear how to select a value for \\(r\\)
in the negative binomial distribution.
In terms of the index of dispersion \\(D =
\frac{\textrm{Var}[y]}{\textrm{E}[y]}\\) and the mean \\(\mu\\), \\(r\\) is given
by[^neg-bin-eta-d]:
\\[ r = \frac{\mu}{D-1} \\]

Assuming that a reasonable choice of \\(r\\) can be selected, the natural
parameter is \\(\eta = \log(1-p)\\), the log-partition function is \\(A(\eta)
= - r \log(1 - e^\eta)\\), the mean is \\(\mu = \frac{rp}{1-p}\\), the canonical
link function is \\(g(\mu) = \log \frac{r}{r+\mu}\\).
With the obvious dispersion parameter \\(\phi = 1\\) the variance function is
\\(V(\mu) = \mu \left( 1 + \frac{\mu}{r} \right)\\).
Since \\(0 < p < 1\\), a non-canonical link function might be useful, such as
the one implied by
\\(\eta = - \exp(-\mathbf{x}^\intercal \boldsymbol{\beta})\\).

[^neg-bin-eta-d]:
    It is a curious fact that the natural parameter \\(\eta\\) is a simple
    function of the index of dispersion: \\[ \eta = -\log D \\]


### Inverse Gaussian

The inverse Gaussian (also known as the Wald) distribution has a somewhat
misleading name -- it is NOT the distribution of the inverse of a
Gaussian-distributed parameter.
Rather it is the distribution of the time it takes for a Gaussian process to
drift to a certain level.
\begin{align}
f(y; \mu, \lambda) &= \sqrt{\frac{\lambda}{2\pi x^3}}
e^{-\frac{\lambda(y-\mu)^2}{2\mu^2 y}} \\\\\
\log f(y; \mu, \lambda) &= -\frac{\lambda(y - \mu)^2}{2\mu^2 y} + \frac{1}{2}
\log \frac{\lambda}{2\pi y^3} \\\\\
&= -\frac{\lambda}{2\mu^2} y - \frac{\lambda}{2} \frac{1}{y} +
\frac{\lambda}{\mu} + \frac{1}{2}\log \lambda + B(y)
\end{align}
The sufficient statistics are therefore \\(\mathbf{T}(y)^\intercal = [y, 1/y]\\)
with corresponding natural parameters \\(\boldsymbol{\eta}^\intercal =
[-\lambda/2\mu^2, -\lambda/2]\\), and the log-partition function is
\\(A(\boldsymbol{\eta}) = 2\sqrt{\eta_1 \eta_2} - \frac{1}{2} \log (-2\eta_2) \\).
The expected values of these statistics are \\(\textrm{E}[\mathbf{T}^\intercal(y); \mu,
\lambda] = [\mu, 1/\mu + 1/\lambda] \\) or \\(\textrm{E}[\mathbf{T}^\intercal
(y); \boldsymbol{\eta}] = \left[\sqrt{\frac{\eta_2}{\eta_1}},
\sqrt{\frac{\eta_1}{\eta_2}} - \frac{1}{2\eta_2}\right]\\).


### Others
Many other distributions are in the exponential family and should be amenable in
principle to a GLM model.
* Chi-squared
* Inverse gamma
* Beta
* Categorical
* Multinomial
* Dirichlet
* Normal-gamma

---
## TODO Summary of GLMs for exponential familes

A summary table would be useful listing the response distribution, natural
parameters, partition function, canonical link function, variance function, etc.
for each common case.

---
## Regularization

Consider the case where the response variable is linearly separable by the
predictor variables in logistic regression. The magnitude of the regression
parameters will increase without bound. This situation can be avoided by
penalizing large values of the parameters in the likelihood function. 

With all forms of regularization the scaling symmetry of each \\(x_k\\) is
broken, so it is likely useful to center and standardize the data.

### L2 regularization (ridge regression)

The parameters can be discouraged from taking very large values by penalizing
the likelihood by their squares.
\\[ l = l_0 - \frac{\lambda_2}{2} \sum_{k=1}^K \left| \beta_{k} \right|^2 \\]
Note that the intercept \\(k=0\\) term is left out of the regularization. Most
regression applications will not want to induce a bias in this term, and there
should be no risk of failing to converge so long as there is more than a single
value in the set of response variables \\(\mathbf{y}\\).

This form of regression is particularly attractive in GLMs because the Jacobian
and Hessian are easily adjusted:
\begin{align}
\mathbf{J} &\rightarrow \mathbf{J} - \lambda_2 \tilde{\boldsymbol{\beta}} \\\\\
\mathbf{H} &\rightarrow \mathbf{H} - \lambda_2 \tilde{\mathbf{I}}
\end{align}
where the tildes are a (sloppy) way of indicating zeros in the \\(0\\) index.
Note that the Hessian is negative-definite, so adding the regularization term
does not risk making it degenerate. In fact, it can help condition for small
eigenvalues of the Hessian.

Another way of denoting the above is with a Tikhonov matrix
\\(\boldsymbol{\Gamma} = \textrm{diag}(0, \sqrt{\lambda_2}, \ldots,
\sqrt{\lambda_2})\\) which penalizes the likelihood by
\\(\left|\boldsymbol{\Gamma}\boldsymbol{\beta}\right|^2/2\\). This makes the
changes to the Jacobian and Hessian easier to express.
\begin{align}
\mathbf{J} &\rightarrow \mathbf{J} - \boldsymbol{\Gamma}^\intercal\boldsymbol{\Gamma}\boldsymbol{\beta} \\\\\
\mathbf{H} &\rightarrow \mathbf{H} - \boldsymbol{\Gamma}^\intercal\boldsymbol{\Gamma}
\end{align}

The IRLS update equation is still quite simple if written in terms of solving a
new guess \\( \boldsymbol{\beta}' \\) in terms of the previous
\\(\boldsymbol{\beta}_\textrm{last}\\).

\begin{equation}
<!-- \left( \mathbf{X}^\intercal \mathbf{S} \mathbf{X} + \boldsymbol{\Gamma}^2 \right) -->
<!-- \boldsymbol{\beta}' = \mathbf{X}^\intercal \left( \mathbf{S} -->
<!-- \mathbf{X} \boldsymbol{\beta}_\textrm{last} + \mathbf{y} - -->
<!-- g^{-1}(\mathbf{X}\boldsymbol{\beta}_\textrm{last}) \right) -->
\end{equation}
Note that the regularization matrix only appears once, as a correction to the
diagonals of the Hessian.

### TODO L1 regularization (lasso)

This could be a bit tricky using IRLS because the Jacobian is non-differentiable
and the Hessian becomes infinite when \\(\beta_k = 0\\) for any \\(k \geq 1\\).
This means that if a parameter is ever at zero exactly (unlikely, except for
starting conditions) then it will never change.

These are solved problems, although they may require heavier machinery.

Parameter selection (choosing which parameters should be zeroed) could perhaps
be combined with the AIC. Such a procedure would be very sensitive to the scale
of the regressor parameters, so they would need to be standardized. In general,
though, solving for the best set of parameters is NP-hard, so heuristics must be used.

### TODO Effective degrees of freedom under regularization

See
[wikipedia](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)#In_non-standard_regression)
and [this paper](https://arxiv.org/abs/1311.2791).

---
## TODO Estimating the real dispersion parameter

Typically the method of moments is used.

See the latter parts of [this lecture on the dispersion parameter](http://people.stat.sfu.ca/~raltman/stat402/402L25.pdf).

---
## TODO Goodness of fit

* Compare log-likelihoods of fit model to saturated model
See [these notes](http://people.stat.sfu.ca/~raltman/stat402/402L11.pdf) on computing the deviance.
* Aikaike and Bayesian information criteria
* Generalized R^2?

---
## TODO Significance of individual parameters

The difference between the likelihood at the fitted values and the likelihood
with one parameter fixed to zero should follow a \\(\chi^2\\) distribution with
1 degree of freedom and implies the \\(Z\\)-score of the parameter. The
likelihood should possibly be re-minimized over the other parameters to allow
them to describe some of what the parameter of interest captured; however, this
could cause to correlated parameters to both appear insignificant when at least
one of them is highly relevant.


---
## TODO Numerical considerations

---
## Other references

* https://www.stat.cmu.edu/~ryantibs/advmethods/notes/glm.pdf
* https://bwlewis.github.io/GLM/
* https://statmath.wu.ac.at/courses/heather_turner/glmCourse_001.pdf
* [Maalouf, M., & Siddiqi, M. (2014). Weighted logistic regression for large-scale imbalanced and rare events data. Knowledge-Based Systems, 59, 142–148.](https://doi.org/10.1016/j.knosys.2014.01.012)
* [Convergence problems in GLMs](https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Marschner.pdf)

---
<!-- Footnotes will appear below the document. -->
Footnotes:
