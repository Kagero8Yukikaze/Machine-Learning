# Representation Theorem (表示定理)

- By default, we use $\Phi(x) \in \mathbb{R}^{d'}$ to replace $x \in \mathbb{R}^d$
- e.g. $\Phi(x)=[x^T,1]^T $, then $f(x)=w^T\Phi(x) $ can recover $w^Tx+b$, which is all linear models' uniform form

## Uniform Form

- 我们之前学过的回归和SVM均可以写成统一的格式
  - Ridge (Linear) Regression
$$
\begin{align*}
    \mathop{min}\limits_{w}\sum_{i=1}^n(w^T\Phi(x_i)-y_i)^2+\lambda||w||^2
\end{align*}
$$

  - Ridge (Logistic) Regression
$$
\begin{align*}
    \mathop{min}\limits_{w}\sum_{i=1}^n\log(1+\exp(-y_iw^T\Phi(x_i)))+\lambda||w||^2
\end{align*}
$$

  - Soft-margin SVM
$$
\begin{align*}
    &\mathop{min}\limits_{w}\frac{1}{2}||w||^2+c\sum_{i=1}^n\max\{0,1-y_iw^T\Phi(x_i)\}\\
    =&\mathop{min}\limits_{w}\sum_{i=1}^n\max\{0,1-y_iw^T\Phi(x_i)\}+\lambda||w||^2
\end{align*}
$$

- 更进一步，我们可以把这种统一的格式改写成一致的核函数的形式
  - Linear Regression
$$
    \begin{align*}
        &\frac{\partial L}{\partial w}=2\sum_{i=1}^n(w^T\Phi(x_i)-y_i)\Phi(x_i)+2\lambda w=0\\
        \Rightarrow &w=\frac{1}{\lambda}\sum_{i=1}^n(y_i-w^T\Phi(x_i))\Phi(x_i):=\sum_{i=1}^n\alpha_i\Phi(x_i)\\
        \Rightarrow &f(x)=\sum_{i=1}^n\alpha_i\Phi^T(x_i)\Phi(x):=\sum_{i=1}^n\alpha_i K(x_i,x)
    \end{align*}
$$

  - Logistic Regression
$$
    \begin{align*}
        &\frac{\partial L}{\partial w}=-\sum_{i=1}^n\frac{\exp(-y_iw^T\Phi(x_i))}{1+\exp(-y_iw^T\Phi(x_i))}y_i\Phi(x_i)+2\lambda w=0 \\
        \Rightarrow &w=\sum_{i=1}^n\frac{1}{2\lambda}\sigma(-y_iw^T\Phi(x_i))y_i\Phi(x_i):= \sum_{i=1}^n\alpha_i\Phi(x_i)\\
        \Rightarrow &f(x)=\sum_{i=1}^n\alpha_i\Phi^T(x_i)\Phi(x):=\sum_{i=1}^n\alpha_i K(x_i,x)
    \end{align*}
$$

  - Soft-margin SVM
    - introduce $\xi_i, \ \ \xi_i \geq 0, \ \ \xi_i\geq 1-y_iw^T\Phi(x)$
    - Dual form
$$
    \begin{align*}
        &\mathop{max}\limits_{\alpha,\beta \geq0}\mathop{min}\limits_{w,\xi}\lambda w^Tw+\sum_{i=1}^n \xi_i +\sum_{i=1}^n \alpha_i(1-y_i w^T\Phi(x_i)-\xi_i)-\sum_{i=1}^n\beta_i\xi_i\\
        \Rightarrow &\begin{split}
            \left\{
                \begin{array}{ll}
                    w=\frac{1}{2\lambda}\sum_{i=1}^n\alpha_i y_i \Phi(x_i):=\sum_{i=1}^n \tilde{\alpha}_i \Phi(x_i) \\
                    \alpha_i+\beta_i=1
                \end{array}
            \right.
        \end{split}\\
        \Rightarrow &f(x)=\sum_{i=1}^n\tilde{\alpha}_i\Phi^T(x_i)\Phi(x):=\sum_{i=1}^n\tilde{\alpha}_i K(x_i,x)
    \end{align*}
$$

## Reproducing Kernel Hilbert Space (再生核希尔伯特空间)

- Vector Space(Euclidean Space $\mathbb{R}^d$) $\rightarrow$ possibly infinite dimensional Hilbert space $\mathcal{H}$(vector space inner product)
  - function $f \in \mathcal{H}$ can be understood as infinite dimensional vector $[f(x_1),\dots,f(x_{\infty})]$
- 该空间满足以下性质:
  - $<f,g>_{\mathcal{H}}=<g,f>_{\mathcal{H}} $
  - $<a_1f_1+a_2f_2,g>_{\mathcal{H}}=a_1<f_1,g>_{\mathcal{H}}+a_2<f_2,g>_{\mathcal{H}} $
  - $<f,f>_{\mathcal{H}} \geq 0, \quad <f,f>_{\mathcal{H}}=0 \Leftrightarrow f=0$
- _RKHS_
  - $f \in \mathcal{H}$, $\mathcal{H}$ is a Hilbert space of real-valued functions $f:X\rightarrow \mathbb{R}$
  - $\mathcal{H}$ is associated with a kernel $K(\cdot,\cdot), \quad s.t. \ \  f(x)=<f,K(\cdot,\cdot)>_{\mathcal{H}} $
  - $K(\cdot,\cdot)$ is called the **reproducing kernel** of RKHS
  - $K$ maps every $x\in X$ to a point in $\mathcal{H}$, $K$ produces $g \in \mathcal{H}$ that $g(z)=K(z,x)$
  - $\mathcal{H}:=\text{Completion of} \{K(\cdot,x)|x\in X \} $
- $<K(z,\cdot),K(\cdot,x)>_{\mathcal{H}}=K(z,x) $
- Example 1
  - $f(x)=w^Tx$ defines $K(z,x)=z^Tx $ is the reproducing kernel of $\mathcal{H}$
  - $f(x)=<K(w,\cdot),K(\cdot,x)>_{\mathcal{H}}=K(w,x)=w^Tx $
- Example 2
  - $f(x)=w^T\Phi(x) $ defines $K(z,x)=\Phi^T(z)\Phi(x) $
  - $f(x)=<K(\Phi^{-1}(w),\cdot),K(\cdot,x)>_{\mathcal{H}}=w^T\Phi(x) $
  - $\Phi$ includes all polynomials that can proximate any functions
- _Most function spaces are RKHS_
- Typically, a function $f:X\rightarrow \mathbb{R}, \ f\in\mathcal{H}$ can be represented as
$$
f(\cdot)=\sum_{i=1}^{\infty}\alpha_iK(x_i,\cdot)
$$

  - $K(x_i,\cdot)$ are basis of $\mathcal{H}$
  - then we have
$$
\begin{align*}
    <f,K(\cdot,x)>_{\mathcal{H}}&=<\sum_{i=1}^{\infty}\alpha_iK(x_i,\cdot),K(\cdot,x)>_{\mathcal{H}}\\
    &=\sum_{i=1}^{\infty}\alpha_iK(x_i,x)=f(x)
\end{align*}
$$

- Norm of $f$
  - $||f||^2_{\mathcal{H}}=<f,f>_{\mathcal{H}}$
  - if $f(x)=w^T\Phi(x)$, then
$$
    \begin{align*}
        <f,f>_{\mathcal{H}}&=<K(\Phi^{-1}(w),\cdot),K(\cdot,\Phi^{-1}(w))>_{\mathcal{H}}\\
        &=K(\Phi^{-1}(w),\Phi^{-1}(w))\\
        &=w^Tw
    \end{align*}
$$
