# Gaussian Process

## Gaussian / Normal Distribution

- Univariate Form
  $$
    \begin{align*}
        X &\sim N(\mu,\sigma^2)\\
        p(x)&=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
    \end{align*}
  $$
- Multivariate Form
  $$
    \begin{align*}
        X&\sim N(X\vert\mu,\Sigma)\\
        X&\in\mathbb{R}^d \quad \mu\in\mathbb{R}^d \quad \Sigma\in\mathbb{R}^{d\times d}\quad\Sigma_{ij}=E(x_i-\mu_i)(x_j-\mu_j)=cov(x_i,x_j)\\
        p(X)&=(\frac{1}{\sqrt{2\pi}})^d \cdot \frac{1}{\vert\Sigma\vert^{\frac{1}{2}}}\cdot\exp(-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu))
    \end{align*}
  $$
  - $\Sigma$ is a covariance matrix, determines how $x_i,x_j$ increase together or not
- CLT
  - When $n$ independent random variables are summed up, their normalized sum tends a _Gaussian distributed random variable_, even if these original random variables are not Gaussian
- Any **linear combination** of Gaussian distributed random variables follow a Gaussian distribution
- **Concatenation** of Gaussian distributed random variables result in a multivariate Gaussian distributed random variable
- Given $x=\begin{pmatrix*}
    x_a\\x_b
\end{pmatrix*}\in\mathbb{R}^{a+b}$, if
  $$
    x\sim N(\begin{pmatrix*}
        \mu_a\\ \mu_b
    \end{pmatrix*},\begin{pmatrix*}
        \Sigma_{aa} & \Sigma_{ab}\\
        \Sigma_{ba} & \Sigma_{bb}
    \end{pmatrix*})
  $$
  then
  1. $x_a \sim N(\mu_a,\Sigma_{aa})$, $x_b \sim N(\mu_b,\Sigma_{bb})$
  2. $P(x_a\vert x_b)=N(x_a \vert \mu_{a\vert b}, \Sigma_{a\vert b})$  
   $\mu_{a\vert b}=\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)$  
   $\Sigma_{a\vert b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$

## Different views of Linear Regression

### ERM view

- the original form:
  $$
    \mathop{min}\limits_{w} \sum_{i=1}^n(y_i-w^T\Phi(x_i))^2
  $$

### MLE view

- $y=w^T\Phi(x)+\epsilon,\quad\epsilon\sim N(0,\sigma^2)$
  - $\epsilon$ is a Gaussian noise
- $P(y\vert x;w,\sigma^2)=N(y|w^T\Phi(x),\sigma^2)$
  - $w$ is the parameter
  - $\sigma^2$ is a hyperparameter
- MLE form:
  $$
    \begin{align*}
        &\mathop{max}\limits_{w}\sum_{i=1}^n\log P(y_i\vert x_i)\\
        \Leftrightarrow&\mathop{max}\limits_{w}\sum_{i=1}^n\log\bigl(\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-w^T\Phi(x_i))^2}{2\sigma^2})\bigr)\\
        \Leftrightarrow&\mathop{max}\limits_{w}\sum_{i=1}^n-\frac{1}{2}\log 2\pi\sigma^2-\frac{(y_i-w^T\Phi(x_i))^2}{2\sigma^2}\\
        \Leftrightarrow&\mathop{min}\limits_{w}\sum_{i=1}^n(y_i-w^T\Phi(x_i))^2
    \end{align*}
  $$

### MAP view

- **Maximum a Posteriori**
  - Previously, we treat $w$ as fixed(constant) parameter. In Bayesian viewpoint, the world is uncertain, even the parameter $w$ are random variables
  - Select the mode of the posteriori distribution as a point estimation
- Prior probability:
  $$
    P(w)=N(w\vert0,\sigma_w^2 I)\quad w\in\mathbb{R}^d
  $$
  - $\mu=0$ is a **prior belief**
  - 我们先验地认为，$w$的期望为0，方差越小，说明我们对$w$的取值在0附近这一事件越自信
- Estimation for _Ridge Regression_:
  $$
    P(y\vert x,w;\sigma^2,\sigma_w^2)=N(y\vert w^T\Phi(x),\sigma^2)
  $$  
  From Bayesian Expectation Equation:
  $$
    P(w\vert y,x)=\frac{P(y\vert x,w)P(w\vert x)}{P(y\vert x)}
  $$
  $P(y|x)$ is not dependent on $w$, let it be $z$, $P(w\vert x)$ is actually $P(w)$, so
  $$
    \begin{align*}
        P(w\vert y,x)&=\frac{1}{z}\cdot\prod_{i=1}^n P(y_i\vert x_i,w)P(w)\\
        &=\frac{1}{z}\cdot(\frac{1}{\sqrt{2\pi}\sigma})^n\cdot\exp(-\frac{\sum_{i=1}^n(y_i-w^T\Phi(x_i))^2}{2\sigma^2})\cdot(\frac{1}{\sqrt{2\pi}\sigma_w})^d\cdot\exp(-\frac{w^Tw}{2\sigma_w^2})
    \end{align*}
  $$
  $$
    \begin{align*}
        \mathop{max}\limits_{w}P(w\vert y,x)&\Leftrightarrow\mathop{max}\limits_{w}-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-w^T\Phi(x_i))^2-\frac{1}{2\sigma_w^2}w^Tw\\
        &\Leftrightarrow \mathop{min}\limits_{w}\sum_{i=1}^n(y_i-w^T\Phi(x_i))^2+\frac{\sigma^2}{\sigma_w^2}\Vert w \Vert^2
    \end{align*}
  $$

## Stochastic (Random) Process

- A collection of (infinitely many) random variables along on index set($\mathbb{N}$ or $\mathbb{R}$ or $\mathbb{R}^d$...)
- Infinitely many $\{(x_i,y_i)\}$ specify a distribution and function $y(x)$. Each sample of $\{(x_i,y_i)\}$ forms a _deterministic function_ $y(x)$
- We can specify a **Random Process** by specifying the **joint distribution** of all random variables

## Gaussian Process (GP)

- We specify the joint distribution over **any finite collection** of variables and require the joint distribution to be Gaussian
- $\{x_1,\dots,x_n\}$ is **any** set of $n$ points in the index set with sampled value $\{y_1,\dots,y_n\}$, then
  $$
    GP \Leftrightarrow y_1,\dots,y_n \text{ have a Multivariate Gaussian Distribution}
  $$
- We need to specify $\mu_i=\text{mean}(x_i)$ and $\Sigma_{ij}=k(x_i,x_j)$ to determine a GP
  $$
    y(x)\sim GP(\text{mean}(x),k(\cdot,\cdot))
  $$
  - we usually use $\text{mean}(x)=0$ (prior belief), which means that $y=w^T\Phi(x)=0$ as $w\sim N(0,\sigma_w^2I)$
  - we only need $k(x_i,x_j)=\exp(-\frac{\Vert x_i-x_j\Vert^2}{2\sigma^2})$
    - $k(x_i,x_j)$ 越大，表明$x_i$和$x_j$越接近，就越有可能同增减
- Suppose $n$ training points $\{x_1,\dots,x_n\}$, let $K$ be Gram matrix, $K_{ij}=k(x_i,x_j)$  
  - $k$ is a valid kernel only if $K\succeq0$ for any $\{x_1,\dots,x_n\}$  
  - Now given a new test point $x^*$, we can compute joint distribution of $\begin{pmatrix*}
    y^*\\y
  \end{pmatrix*}\in\mathbb{R}^{n+1}$
  $$
    \begin{align*}
        P(\begin{pmatrix*}
            y^*\\y
        \end{pmatrix*})&=N(y\vert0,\begin{pmatrix*}
            k(x^*,x^*)&k^T(x^*)\\
            k(x^*)&K
        \end{pmatrix*}), \quad k(x^*)=\begin{pmatrix*}
            k(x^*,x_1)\\ \vdots \\ k(x^*,x_n)
        \end{pmatrix*}\\
        P(y^*\vert y)&=N(y^*\vert \mu^*,\Sigma^*)\\
        \mu^*&=0+k^T(x^*)K^{-1}(y-0)=k^T(x^*)K^{-1}y\\
        \Sigma^*&=k(x^*,x^*)-k^T(x^*)K^{-1}k(x^*)
    \end{align*}
  $$
  So we have the dual form of Linear Regression:
  $$
    f(x^*)=\mu^*=k^T(x^*)K^{-1}y
  $$
- In a more realistic setting, we can only observe $\hat{y}=y+\epsilon,\quad \epsilon\sim N(0,\sigma^2I)$, then
  $$
    \begin{align*}
        P(\hat{y})&=N(y\vert 0,K+\sigma^2I)\\
        \mu^*&=k^T(x^*)(K+\sigma^2I)^{-1}y\\
        \Sigma^*&=k(x^*,x^*)-k^T(x^*)(K+\sigma^2I)^{-1}k(x^*)
    \end{align*}
  $$
  because
  $$
    \begin{align*}
        \text{cov}(\hat{y_i},\hat{y_j})&=E\hat{y_i}\hat{y_j}-E\hat{y_i}E\hat{y_j}\\
        &=E(y_i+\epsilon_i)(y_j+\epsilon_j)\\
        &=Ey_iy_j+E\epsilon_i\epsilon_j+Ey_i\epsilon_j+Ey_j\epsilon_i\\
        &=Ey_iy_j+E\epsilon_i\epsilon_j\\
        &=\text{cov}(y_i,y_j)+\text{cov}(\epsilon_i,\epsilon_j)\\
        &=k(x_i,x_j)+\sigma^2 1(i=j)
    \end{align*}
  $$
