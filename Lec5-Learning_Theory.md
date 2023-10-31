# Learning Theory

## PAC framework

- Train a model $\Leftrightarrow$ Take a course
- Training examples $\Leftrightarrow$ Take exercises/homework
- Testing $\Leftrightarrow$ Take exam
- Can we estimate our performance in exam by performance on exercise?
  - training error $\rightarrow$ testing error
- $E_{in} :=$ the in-sample error i.e. **Error in Training data**
  - Let $h\in\mathcal{H}$
    - e.g. $h(x)=\text{sign}(w^Tx+b)$
  - model function hypothesis $\rightarrow$ hypothesis space
  - $E_{in}(h)=\frac{1}{n}\sum_{i=1}^n1(h(x_i)\ne y_i) \rightarrow $ **Error Rate**
  - $\{(x_1,y_1),\dots,(x_n,y_,)\}$ are training data, $(x_i,y_i)\sim P_{xy}$
- $E_{out} :=$ the out-of-sample error
  - measures how well a model **generalizes**
  - $E_{out}(h):=P(h(x)\ne y)=E_{(x,y)\sim P_{xy}}[1(h(x)\ne y)] $
- $E_{out}(h)-E_{in}(h) $ is the **Generalization Error**
- We can say with a large probability $1-\delta$ ($\delta$ is small), $E_{out}(h)-E_{in}(h)< \delta$. This is called **Probably Approximately Correct (PAC) Learning Framework**

## Hoeffding Inequality

- $x_1,\dots,x_n$ are independent random variables, $x_i \in [a_i,b_i]$, $\bar{x}=\frac{1}{n}\sum_{i=1}^nx_i$. Then $\forall \epsilon > 0$, we have
  - $P(\bar{x}-E[\bar{x}]\geq\delta)\leq\exp(-\frac{2n^2\epsilon^2}{\sum_{i=1}^n(b_i-a_i)^2})$
  - $P(E[\bar{x}]-\bar{x}\geq\delta)\leq\exp(-\frac{2n^2\epsilon^2}{\sum_{i=1}^n(b_i-a_i)^2})$

## Growth Function

- Now for a given fixed $h$, we have:  
  $P(E_{out}(h)-E_{in}(h)\geq\epsilon)\leq\exp(-2n\epsilon^2)$  
  Because
  $$
    \begin{align*}
        E_{in}(h)&=\frac{1}{n}\sum_{i=1}^n1(h(x_i)\ne y_i)=\bar{x} \quad \Rightarrow x_i \in [0,1]\\
        E[\bar{x}]&=E[\frac{1}{n}\sum_{i=1}^n1(h(x_i)\ne y_i)]\\
        &=\frac{1}{n}\sum_{i=1}^nE_{(x,y)\sim P_{xy}}[1(h(x_i)\ne y_i)]\\
        &=\frac{1}{n}nE_{out}(h)\\
        &=E_{out}(h)\\
        P(E_{out}(h)&-E_{in}(h)<\epsilon)>1-\exp(2n\epsilon^2)
    \end{align*}
  $$

  - with probability at least $1-\delta$, $\exist h\in\mathcal{H}$, $E_{out}(h)-E_{in}(h)<\epsilon $
  - But this bound doesn't consider training by assuming $h$ is given before _seeing_ the training data. It's not meaningful in practice.

- Since we cannot know which $h\in\mathcal{H}$ to use before seeing training data, we can bound $\mathcal{H}$ instead, thus independent of particular $h$.  
  Let's first assume $\mathcal{H}$ is finite, $\mathcal{H}=\{h_1,\dots,h_M\}$.
  $$
    \begin{align*}
        P(\exist h\in\mathcal{H},E_{out}(h)-E_{in}(h)\geq\epsilon)&\leq\sum_{i=1}^nP(E_{out}(h_i)-E_{in}(h_i)\geq\epsilon)\\
        &\leq M\exp(-2n\epsilon^2)
    \end{align*}
  $$
  This is the first practical **PAC learning bound**.  
  Let
  $$
    \begin{align*}
        \delta&=M\exp(-2n\epsilon^2)\\
        \epsilon&=\sqrt{\frac{1}{2n}\log\frac{M}{\delta}}
    \end{align*}
  $$
  With probability at least $1-\delta$, we have $\forall h\in\mathcal{H}$, $E_{out}(h)-E_{in}(h)<\sqrt{\frac{1}{2n}\log\frac{M}{\delta}} $
  - $n\nearrow, \ M\searrow,$ generalization error $\searrow$
  - $M\nearrow$, will be overfitting
- What if $H$ is infinite?
  - These different hypotheses(hyperplane) give the same classification results on finite examples.
  - Union bound counts each $h$ once, but if one $h$ satisfies **PAC learning bound**, then all other $h$ (with same classification results) will also satisfy the PAC bound.
- Growth Function
  - measure effective $\#$ of hypotheses in $\mathcal{H}$ on finite data
  - Assume binary classification $y\in\{-1,1\}$, $x\in X$  
    Given $n$ training samples $x_1,\dots,x_n \in X$, apply $h\in\mathcal{H}$ to them to get $n$-tople ($h(x_1),\dots,h(x_n)$) of $\plusmn1$s. (called a _dichotomy_)  
    Let $\mathcal{H}(x_1,\dots,x_n)=\{\bigl(h(x_1),\dots,h(x_n)\bigr)\vert h\in\mathcal{H}\}$. (set has no repeated elements)  
    then $m_{\mathcal{H}}(h):=\mathop{max}\limits_{x_1,\dots,x_n\in X}\vert \mathcal{H}(x_1,\dots,x_n) \vert$