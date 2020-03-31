---
layout: post
title:  "Mathematics of Generalized Linear Models"
date:   2020-03-30 19:10:30 -0500
permalink: "glm-math"
categories: statistics
tags: [glm, fitting, generalized linear model]
mathjax: true
---

# {{page.title}}

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

NOTE: Observations can be considered with different weights by given each a
different dispersion parameter, which otherwise does not affect the calculations above.

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
\boldsymbol{\eta}''\right]
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
\textrm{Var}[y^{(i)}|\mathbf{x}^{(i)} \cdot \boldsymbol{\beta}]\\). Correlations
between different observations could be induced by allowing off-diagonal terms
in \\(\mathbf{S}\\) (TODO: work out how to do this consistently).

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
\\(y\\), this update rule becomes the following.
\begin{equation}
\left( \mathbf{X}^\intercal \mathbf{S} \mathbf{X} \right) \cdot \Delta
\boldsymbol{\beta} = \mathbf{X}^\intercal \cdot \left[ \mathbf{y} -
g^{-1}(\mathbf{X}\boldsymbol{\beta})\right]
\end{equation}

## TODO Case studies

### Ordinary least squares

### Logistic

### Poisson

### Exponential
\\[ f(y; \lambda) = \begin{cases}
\lambda e^{-\lambda y} & \textrm{for } y \geq 0 \\\\\
0 & \textrm{for } y < 0
\end{cases}\\]
The exponential distribution is a special case of the gamma distribution with
\\(\alpha = 1\\).

### Binomial with known \\(n\\)

### Gamma
The gamma distribution has two parameters, but it turns out that the shape
parameter \\(\alpha\\) is often treated as the same for every observation, allowing
for the \\(\beta\\) parameter to be predicted by
\\(\mathbf{x}\cdot\boldsymbol{\beta}\\). <a href="https://civil.colorado.edu/~balajir/CVEN6833/lectures/GammaGLM-01.pdf">See these notes</a>.
This seems analogous to the situations in OLS where the variance \\(\sigma^2\\)
is the same for each data point so the minimization is unaffected by its value.

### Negative binomial

#### Known \\(r\\)

#### Unknown \\(r\\)

## TODO Regularization

## TODO Goodness of fit
Compare log-likelihoods of fit model to saturated model

## TODO Numerical considerations
