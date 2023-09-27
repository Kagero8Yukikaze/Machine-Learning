# Support Vector Machine (SVM)

- Binary Classification with linear models
- $y \in \{-1,1\} $ and $x \in \mathbb{R}^d$

## Constraint Optimization

### Lagrange function

- $L(x,\lambda):=f(x)+\lambda h(x)$
  - $\lambda$ is _Lagrange Multiplier_


### Equality Constraint

- $\mathop{min}\limits_{x}f(x) \ \ \  s.t. \ \  h(x)=0$
  - $\forall$ point $x$ on the constraint surface $h(x)=0$, we have $\nabla h(x)$ orthogonal to the surface
    - Because if $\nabla h(x)$ has tangent component, we can move along the direction to make $h(x) \neq 0$
  - For a local minimum $x^*$, $\nabla f(x^*)$ must be orthogonal to the surface
    - Generally, $x^*$ is a local minimum $\Rightarrow$ $\exist \lambda \ \ \ s.t. \ \nabla f(x^*)+\lambda \nabla h(x^*)=0$
    - (There may be some corner cases not satisfying this equation)
  - 可以说，极值点往往是限制平面与函数相切的点，那么此时该点处平面的梯度与函数的梯度就应该是相反的
- When we have $K$ constraints:
  - $\mathop{min}\limits_{x} f(x) \ \ \ s.t. \ \ h_i(x)=0, \ i \in \{1,\dots,K\}$
  - then $L(x,\lambda) = f(x)+\sum_{i=1}^K \lambda_i h_i(x) $
- now we can say that:
$$
x^* \ is \ a \ local \ minimum \Rightarrow \exist \lambda \left\{
    \begin{array}{ll}
        \nabla_x L(x^*,\lambda) = 0 & (1)\\
        \nabla_{\lambda}L(x^*,\lambda) = 0 & (2)
    \end{array}
\right.
$$

  - $(1)$说明 $\lambda f(x^*)+\sum_{i=1}^n \lambda_i \nabla h_i(x^{ *})=0$
  - $(2)$说明 $h_i(x^{ * })=0, \ \ \forall i$

### Inequality Constraint

- $\mathop{min}\limits_{x}f(x) \ \ \  s.t. \ \  g(x) \leq 0$
  - $\forall$ point $x$ on the surface $g(x)=0$, $\nabla g(x)$ must be orthogonal to the surface and _point out of_ the region of $g(x) \leq 0$
  - For a local minimum $x^*$
    - the constraint is _active_:
      - if $x^*$ on the surface $g(x)=0$, then $-\nabla f(x^*)$ must have the same direction as $\nabla g(x^*)$
      - $\exist \mu > 0, \ \ s.t. \ \nabla f(x^*)+\mu \nabla g(x^*)=0 $
    - the constraint is _inactive_:
      - if $x^*$ within the surface, we only need $\nabla f(x^*)=0$
      - $\exist \mu = 0, \ \ s.t. \ \nabla f(x^*)+\mu \nabla g(x^*)=0 $
    - 显然地，如果函数的最小值点在平面内部，自然这个平面的约束就没有用了，极值点就是这个最小值点；而如果函数的最小值点在平面外，那么就和equality constraint一样，极值点在平面的边界上
    - 总结:
$$
\exist \mu \geq 0, \ \ s.t. \ \nabla f(x^*)+\mu \nabla g(x^*)=0
\left\{
    \begin{array}{ll}
        \mu = 0 & inactive\\
        \mu > 0 & active
    \end{array}
\right.
$$

### K.K.T conditions

- $\mathop{min}\limits_{x}f(x) \ \ \  s.t. \ \  h_i(x)=0, \ i\in \{1,\dots,K\}, \ \ g_j(x) \leq 0, \ j \in \{1,\dots,L\}$
- $L(x,\lambda,\mu) = f(x)+\sum_{i=1}^K \lambda_i h_i(x) + \sum_{j=1}^L \mu_j g_j(x)$
- **K.K.T conditions**:
$$
    x^* \ is \ a \ local \ minimum \Rightarrow \left\{
        \begin{array}{ll}
            \nabla_x L(x^*,\lambda,\mu)=0 & \\
            h_i(x^*)=0 & \forall i \\
            g_j(x^*)\leq0 & \forall j \\
            \mu_j \geq 0 & \forall j \\
            \mu_j g_j(x^*)=0 & \forall j
        \end{array}
    \right.
$$

  - 对于$\mu_j g_j(x^*)=0$，若$x^*$在边界处，显然有$g_j(x^*)=0$；若$x^*$在内部，则有$\mu=0$，所以该式恒等于0

## Maximum Margin Criterion

- Maximize margin (minimum distance to the hyperplane $w^Tx+b=0$, over all training data)
  - _Structural Risk Minimization_
- $x$到$w^Tx+b=0$的距离为$\Delta x = \frac{w^Tx+b}{||w||}$
- we can define $\gamma_i := y_i \Delta x_i = \frac{y_i(w^Tx+b)}{||w||} \geq 0$ (as $y_i \in \{-1,1\}$)
- $\gamma = \mathop{min}\limits_{i=1,\dots,n} \gamma_i = \mathop{min}\limits_{i=1,\dots,n} \frac{y_i(w^Tx+b)}{||w||} $
  - this is called **geometric margin**
- now we need to maximize this distance:
$$
    \mathop{max}\limits_{w,b,\gamma}\gamma \ \ \ s.t. \frac{y_i(w^Tx_i+b)}{||w||} \geq \gamma, \ \ \forall i
$$

  - 但显然三个参数对于我们来说还是太多了，所以我们可以先假设找到了离超平面$w^Tx+b=0$最近的点$x_0$，此时有$\gamma_0 = \frac{y_0(w^Tx_0+b)}{||w||} = \gamma$
- then we can update our object:
$$
    \mathop{max}\limits_{w,b}\frac{y_0(w^Tx_0+b)}{||w||} \ \ \ s.t. \ \  y_i(w^Tx_i+b) \geq y_0(w^Tx_0+b), \ \ \forall i
$$

  - 我们可以发现，$w^Tx+b=0$是一个超平面，而$kw^Tx+kb=0, \ \forall k$都是同一个超平面，那么这样的话$ \frac{y_0(w^Tx_0+b)}{||w||} $就可以是任意值(which is called **functional margin**)，所以我们可以干脆令$y_0(w^Tx_0+b)=1$，背后的逻辑是不管真实的$\gamma$是多少，我都可以找到一个k使上式等于1，而且显然这个等式也是不影响$\gamma$的
- then we can say that:
$$
    \mathop{max}\limits_{w,b}\frac{1}{||w||} \ \ \ s.t. \ \  y_i(w^Tx_i+b) \geq 1, \ \ \forall i
$$

- now we have the **primal form** of SVM:
$$
    \mathop{min}\limits_{w,b}\frac{1}{2}w^Tw \ \ \ s.t. \ \  y_i(w^Tx_i+b) \geq 1, \ \ \forall i
$$
