
# Mini-batch gradient descent

- With the implementation of gradient descent on the whole training set, you have to process your entire training set before you take one little step of gradient descent.
- Vectorization allows you to efficiently compute on m examples. But m if is really large that can still be slow. 

#### A solution to this is only ingest a small fixed amount of examples (1000, for example) and use each one to iteratively compute the errors.

#### Now, mini batch number T is going to be comprised of (XT, YT).

> doing one epoch of training and epoch is a word that means a single pass through the training set.
---

# Understanding mini-batch gradient descent

### If you plot progress on your cost function,
- With the batch the cost should decrease on every iteration, On mini batch gradient descent though, it may not decrease on every iteration.
- With the mini batch you’re using just a small sample of the data, it should trend downwards, but it's also going to be a little bit noisier.

### On both extremes 
- **If size = m :**
    - Then you just end up with batch gradient descent.
    - With size = m it takes too long but the advantages is that the convergence has much less noise


- **If size = 1 :**
    - This gives you an algorithm called **stochastic gradient descent**. Where, every example is its own mini-batch.
    - Stochastic gradient descent can be extremely noisy, and also loses the advantages of vectorization.
    - Stochastic gradient descent won't ever converge, it'll always just kind of oscillate and wander around the region of the minimum.


### In practice size must be between 1 and m to ensure that it really converges
- The in-between has both advantages (vectorization on memory and quick conversion) with reduced disadvantages

### But how do I choose my batch size? 
- If the training set is small use it all. 
- If it’s bigger use a mini-batch size closest to a given n where size = 2^n.

### The noisiness can be reduced by just using a smaller learning rate.
---

# Exponentially weighted averages

` They are used in other optimization algorithms that are faster than gradient descent`


<img src="https://i.ytimg.com/vi/NxTFlzBjS-4/maxresdefault.jpg" width="550">


- Small **β** values:
    - shorter windows of influence,
    - quick response regressions 
    - much noiser curves 
    - much more susceptible to outliers.

- Long **β** values:
    - longer windows of influence
    - delayed regressions
    - smoother curves

---
# Understanding exponentially weighted averages


- The impact of beta changes much simply because the equation is recursive. Recursively the importance of the **β** value decreases exponentially
- Implementing the recursive code we don’t really care for all the **V**’s, just the current day **V** value
- One of the advantages of this exponentially weighted average formula, is that it takes very little memory.

### For Example

The way you compute **V100**, is you take the **element wise product** between the parameter values and the exponentially decaying function, and then summing it up.

### Pseudo-code

```
Vθ = 0

Repeat {

Get next θt

Vθ = β Vθ + (1 - β) θt

}
```

---

# Bias correction in exponentially weighted averages

```
Bias correction can make you computation of these exponentially weighted averages more accurately.
```

<img src="https://wikidocs.net/images/page/35847/04_020.png" width="600">

---

# Gradient descent with momentum

```
Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has 
some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence.
Using momentum can reduce these oscillations. And Gradient descent with momentum is almost always faster than normal Gradient descent.
```

### Steps:

- compute an exponentially weighted average of the gradient
- use that gradient to update the weights instead.

#### This up and down oscillations slows down gradient descent and prevents you from using a much larger learning rate bcz you might end up over shooting and end up diverging like so.

#### And so, you want to slow down the learning in the b direction, or in the vertical direction. And speed up learning, or at least not slow it down in the horizontal direction.


<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image018.png">

## GD with momentum Pseudo-code

- 1) computing the moving average of the derivatives for w and b.
- 2) And then you would update W and b, instead of updating it with the derivative dW and db, you update it with **vdW** and **vdb**.


<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image019.png" width="600">


- **In practice:**
    - **β=0.9** works well for most cases
    - no bias correction implemented
    - can use **β=0** → ~VdW is scaled by **1/(1-β)** → use a scaled alpha then.

---

# RMSprop ("Root-Mean-Square-prop")

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image022.png" width="500">


```
Computing an exponentially weighted average of the squares of the derivatives, and the squaring is an element-wise operation.
```

- What if square root of **SdW**, right, is very close to **0**. Then things could blow up. Just to ensure numerical stability, when you implement this in practice you add a very, very small epsilon to the denominator.


---

# Adam optimization algorithm

- A combination of **RMSprop** and **momentum**. Proved to work well for a varity of problems.

- **Step:**
    - Maintain both VdW, Vdb (hyperparam=β1) and SdW, Sdb (hyperparam=β2)
    - Implement bias correction: V_corrected, S_corrected divided by (1-β^t)
    - Param update (W, b): `V / sqrt(S)`

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image024.png" width="600">


- **hyperparameters:**

    - **alpha**: learning rate, needs tuning
    - **β1**: usually 0.9
    - **β2**: usually 0.999
    - **epsilon (ε)**: 10e-8 (not important)


- **Some advantages of Adam include:**
    - Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
    - Usually works well even with little tuning of hyperparameters (except  αα )

---
# Learning rate decay

```
One of the things that might help speed up your learning algorithm, is to slowly reduce your learning rate over time.
We call this learning rate decay.
```
<img src="https://upscfever.com/upsc-fever/en/data/deeplearning2/images/learning-rate-decay.png" width="600">


## Implementation:

- 1 epoch = 1 pass through whole data.
- decay learning rate alpha after each epoch:
- Note that the decay rate and  here becomes another hyper-parameter, which you might need to tune.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image028.png" width="500">

## other decay method:

### 1) exponentially decay alpha:

- Where **alpha** is equal to some number less than 1, such as **0.95** times epoch-num, times alpha 0.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image029.png">

### 2) sqrt of epoch_num:

- alpha = some constant / epoch-num square root times alpha 0.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image030.png">

### 3) discrete staircase:

- A learning rate that decreases in discrete steps. Wherefore some number of steps, you have some learning rate, and then after a while you decrease it by one half. After a while by one half. After a while by one half. And so this is a discrete staircase.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image031.png">

### 4) Manual decay:

- If you're training just one model at a time, and if your model takes many hours, or even many days to train. What some people will do, is just watch your model as it's training over a large number of days.
- This works only if you're training only a small number of models, and when training takes long time.


---

# The problem of local optima

```
In the picture below, it looks like there are a lot of local optimas. And it would be easy for GD, or one of the 
other algorithmsto get stuck in a local optimum rather than find its way to a global optimum.
```

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image032.png">

## Saddle points

```
It turns out if you create a neural network, most points of zero gradients are not the local optima,
Instead most points of zero gradient in a cost function are saddle points.
In very high-dimensional spaces you're actually much more likely to run into a saddle point
```

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image034.png" width="500">

## Plateau: region where gradient close to 0 for long time.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk2/pasted_image035.png" width="500">

### Take-away:
- unlikely to stuck in bad local optima: D dimentional → ~2^(-D) of chance.
- plateaus can make learning slow → use momentum/RMSprop/Adam to speedup training.

---


2- What’s new for you ?
---

## ALL


3- Resources ? 
---
- [https://upscfever.com/upsc-fever/en/data/deeplearning2/18.html](https://upscfever.com/upsc-fever/en/data/deeplearning2/18.html)
- [https://x-wei.github.io/Ng_DLMooc_c2wk2.html]
