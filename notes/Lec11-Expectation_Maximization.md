# Expectation Maximization

## EM in general

- Suppose $\theta$ contains all parameters to estimate
- In MLE, we are to maximize $P(x;\theta)$ (evidence) on training data $D=\{x_1,\dots,x_n\}$. We assume directly optimizing $P(x;\theta)$ is difficult, but optimizing $P(x,z;\theta)$ is easy
  $$
    \begin{align*}
        P(x,z)&=P(z)P(x \vert z)\\
        P(x)&=\sum_z P(x,z)\\
        P(z \vert x)&=\frac{P(z)P(x \vert z)}{\sum_z P(z)P(x \vert z)}
    \end{align*}
  $$
  However, $\sum_z P(z)P(x \vert z)$ is _intractable_ due to sum in denominator  
  Find a **variational distribution** $q(z\vert x)$ (which is _tractable_) to approximate $P(z\vert x)$
- Now we introduce $q(z\vert x)$ to approximate $P(z\vert x; \theta)$  
  We have the following always holds:
  $$
    \begin{align*}
        \log P(x;\theta) &= \sum_z q(z\vert x)\log P(x;\theta)\\
        &=\sum_z q(z\vert x)\bigl(\log P(x,z;\theta)-\log P(z\vert x;\theta)\bigr)\\
        &=\sum_z q(z\vert x)\Bigl(\bigl(\log P(x,z;\theta)-\log q(z\vert x)\bigr)-\bigl(\log P(z\vert x;\theta)-\log q(z\vert x)\bigr)\Bigr)\\
        &=\sum_z q(z\vert x)\log\frac{P(x,z;\theta)}{q(z\vert x)}-\sum_z q(z\vert x)\log\frac{P(z\vert x;\theta)}{q(z\vert x)}
    \end{align*}
  $$
  - first half is called **Evidence Lower Bound (ELBO)**
  - second half is called **KL divergence** or KL$(q \Vert p)$
    - measures divergence of $p$ and $q$
    - $\sum_z q(z\vert x)\log\frac{P(z\vert x;\theta)}{q(z\vert x)}\leq\log\sum_z q(z\vert x)\frac{P(z\vert x;\theta)}{q(z\vert x)}=0$ (Jensen's Inequality)  
    $\Rightarrow \text{KL}(q\Vert p)\geq0$, when $q=p$, takes 0
- EM is maximizing ELBO. Maximizing the lower bound also increase evidence

## EM steps

- EM is a two-step iterative algorithm to _maximize ELBO_
  $$
    L(q,\theta) = \sum_z q(z\vert x)\log\frac{P(x,z;\theta)}{q(z\vert x)}
  $$

1. **E-step**: Given $\theta$ fixed, optimize $q$  
   Let's assume current $\theta = \theta^{old}$, ELBO $L(q,\theta^{old})$ is functional of $q$  
   Also when $\theta^{old}$ is fixed, $\log P(x;\theta^{old})$ is a _constant_  
   To maximize ELBO w.r.t $q \quad\Rightarrow\quad \mathop{min}\text{KL}(q\Vert p)\quad\Rightarrow\quad q(z\vert x)=P(z\vert x;\theta^{old})$  
   That is, E-step just let $q(z\vert x)$ take the **posterior** $P(z\vert x;\theta^{old})$
2. **M-step**: Given $q$ fixed, optimize $\theta$  
   Now we have $q(z\vert x)=P(z\vert x;\theta^{old})$. Substitute into ELBO:
   $$
    \begin{align*}
        L(q,\theta)&=\sum_zP(z\vert x;\theta^{old})\log\frac{P(x,z;\theta)}{P(z\vert x;\theta^{old})}\\
        &=\sum_z P(z\vert x;\theta^{old})\log P(x,z;\theta)-\text{const}\\
        \Rightarrow&\mathop{max}\limits_{\theta}E_z\log P(x,z;\theta)\quad z\sim P(z\vert x;\theta^{old})
    \end{align*}
   $$
3. Repeat 1 and 2 until convergence

## EM for GMM

- Objective:
  $$
    \log P(x;\theta) = \sum_{i=1}^n\log\bigl(\sum_{k=1}^K\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)
  $$
- **E-step** compute $q(z\vert x)=P(z\vert x;\mu,\Sigma,\pi)$  
  Let $z_{ik}=\{0,1\}$ indicates $x_i$ whether generated from cluster $k$
  $$
    \begin{align*}
        P(z,x;\mu,\Sigma,\pi)&=\prod_{i=1}^nP(x_i\vert z_i; \mu,\Sigma) P(z_i;\pi)\\
        &=\prod_{i=1}^n\prod_{k=1}^K N(x_i\vert\mu_k,\Sigma_k)^{z_{ik}} \prod_{k=1}^K \pi_k^{z_{ik}}\\
        &=\prod_{i=1}^n\prod_{k=1}^K\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)^{z_{ik}}\\
        P(z\vert x;\mu,\Sigma,\pi)&=\frac{1}{P(x)}\prod_{i=1}^n\prod_{k=1}^K\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)^{z_{ik}}\\
        &=\prod_{i=1}^n\Bigl(\frac{1}{P(x_i)}\prod_{k=1}^K\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)^{z_{ik}}\Bigr)\\
        &=\prod_{i=1}^n P(z_i\vert x_i; \mu,\Sigma,\pi)
    \end{align*}
  $$
- **M-step**  
  Objective:
  $$
    \begin{align*}
        \mathop{max}\limits_{\mu,\Sigma,\pi} &E_z\bigl(\log P(x,z;\mu,\Sigma,\pi) \bigr)\quad z\sim P(z\vert x;\mu^{old},\Sigma^{old},\pi^{old})\\
        =&E_z\Bigl( z_{ik}\log\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr) \Bigr)\\
        =&\sum_{i=1}^n\sum_{k=1}^KE_z(z_{ik})\log\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)
    \end{align*}\\
  $$
  We can compute $E_z(z_{ik})$
  $$
    \begin{align*}
        E_z(z_{ik})&=1\cdot P(z_{ik}=1\vert x_i) + 0\cdot P(z_{ik}=0\vert x_i)\\
        &=\frac{P(z_{ik}=1, x_i)}{P(x_i)}\\
        &=\frac{\pi_k N(x_i\vert \mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j N(x_i\vert \mu_k,\Sigma_k)} = \gamma_{ik}
    \end{align*}
  $$
  Update the objective:
  $$
    \mathop{max}\limits_{\mu,\Sigma,\pi} \sum_{i=1}^n\sum_{k=1}^K\gamma_{ik}\log\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)\quad \text{s.t.} \sum_{k=1}^K \pi_k=1
  $$
  Use Lagrange function
  $$
    L(\mu,\Sigma,\pi,\lambda)=\sum_{i=1}^n\sum_{k=1}^K\gamma_{ik}\log\bigl(\pi_k N(x_i\vert \mu_k,\Sigma_k)\bigr)+\lambda(1-\sum_{k=1}^K \pi_k)
  $$
  - Compute $\pi_k$
    $$
    \begin{align*}
        \frac{\partial L}{\partial \pi_k}&=\sum_{i=1}^n\frac{\gamma_{ik}}{\pi_k}-\lambda=0\\
        \pi_k&=\frac{\sum_{i=1}^n\gamma_{ik}}{\lambda}\\
        \lambda \sum_{k=1}^K\pi_k&=\sum_{i=1}^n\sum_{k=1}^K\gamma_{ik}=n \ \ \Rightarrow \ \ \lambda =n\\
        \pi_k&=\frac{\sum_{i=1}^n\gamma_{ik}}{n}
    \end{align*}
    $$
  - Compute $\mu_k$ and $\Sigma_k$  
    We know that
    $$
    N(x_i\vert \mu_k,\Sigma_k)=\frac{1}{(2\pi)^{\frac{d}{2}}\vert\Sigma_k\vert ^{\frac{1}{2}}}\exp\bigl(-\frac{1}{2}(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k)\bigr)
    $$
    So
    $$
    \begin{align*}
         L(\mu,\Sigma,\pi,\lambda)&=\sum_{i=1}^n\sum_{k=1}^K\gamma_{ik} \bigl(\log\pi_k-\frac{d}{2}\log2\pi-\frac{1}{2}\log\vert \Sigma_k\vert-\frac{1}{2}(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k) \bigr)\\
        \frac{\partial L}{\partial \mu_k}&=\sum_{i=1}^n \gamma_{ik}\Sigma_k^{-1}(x_i-\mu_k)=0\\
        \mu_k&=\frac{\sum_{i=1}^n\gamma_{ik}x_i}{\sum_{i=1}^n\gamma_{ik}}\\
        \frac{\partial L}{\partial \Sigma_k}&=\sum_{i=1}^n \gamma_{ik}\frac{1}{\vert\Sigma_k\vert} \frac{\partial\vert\Sigma_k\vert}{\partial \Sigma_k}+\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T\\
        &=-\sum_{i=1}^n \gamma_{ik}\Sigma_k+\sum_{i=1}^n\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T=0\\
        \Sigma_k&=\frac{\sum_{i=1}^n\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{\sum_{i=1}^n\gamma_{ik}}
    \end{align*}
    $$
