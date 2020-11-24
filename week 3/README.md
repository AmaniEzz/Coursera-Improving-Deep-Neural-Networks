
# Tuning process

## Hyperparameters to deal with in deep learning (in order of importance) 

- 1) Learning rate **alpha** (most importance)
- 2) momentum term **beta (β)**.
- 3) Mini-batch size.
- 4) No. of hidden units.
- 5) No. of Layers.
- 6) Learning rate decay.
- 7) **β1=0.9**, **β2=0.999** and **ε=10e-8** for Adam Optimaization.


## 1- Try random values, don't use a grid search (grid of n * n).

- In grid search: only n distinct values of alpha are tried
- In random choice: can have n*n distinct values of alpha

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image001.png" width="550">

## 2- Coarse to fine sampling scheme.

```
Zoom in to smaller regions of hyperparam space and re-sample more densely.
```

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image002.png" width="550">

---

# Using an appropriate scale to pick hyperparameters

- "Sampling at random", but at appropriate scale, not uniformly.

### Example: choice of alpha in [0.001, 1]

- → sample uniformly at log scale is more resonable: equal resources are used to search at each scale.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image003.png">

### Implementation:

- → to sample on the log scale, you take the low value, take logs to figure out what is a.
- → So now you're trying to sample, from 10 to the a to the b, on a log scale.
- → So you set r uniformly, at random, between a and b.
- → And then you set the hyperparameter to be 10 to the r.

```python
r = -4 * np.random.rand()  # -4 <= r <= 0, uniformly at randome  
alpha = np.exp(10, r) # 10e-4 <= alpha <= 1.0
```

## Sampling β for exp-weighted-avg:
### Example: sample in the range of [0.9, 0.999]

- → convert it to be sampling over (1-β), which is in range [0.1, 0.0001]
- → sample **r** uniformly, at random, from [-3 to -1].
- → and you set `1-β = 10^r`, 
- → and so `β = 1-10^r`.

> Why linear scaling is a bad idea?
>> because that formula we have, `1 / 1- β`, this is very sensitive to small changes in beta, when beta is close to 1.
>> So what this whole sampling process does, is it causes you to sample more densely in the region of when beta is close to 1. Or, alternatively, when 1- beta is close to 0.
---

# Hyperparameters tuning in practice: Pandas vs. Caviar

Tricks on how to organize hyperparam tuning process.

## Two major schools of training

### 1) Panda approach: babysitting one model

```
babysitting the model one day at a time even as it's training over a course of many days or over the course of several different weeks.
And every day you kind of look at it and try nudging up and down your parameters.
```

- It is used when:
    - we have Huge dataset
    - we have limited computing resources, can only train one model.

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image004.png">

### 2) Caviar approach: train many models in parallel

```
this way you can try a lot of different hyperparameter settings and then just maybe quickly at the end pick the one that works best.
this method is used when we have enough computation power.
```

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image005.png">

---

# Normalizing activations in a network

```
- Batch normalization makes your hyperparameter search problem much easier, makes your neural network much more robust.
- The choice of hyperparameters that work well is in a much bigger range,
- and will also enable you to much more easily train even very deep networks.
```


### The question is :

→ For any hidden layer `a`, can we normalize, The values of `a[l-1]`, so as to train `W[l]` and `b[l]` faster??

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image011.png" width="600">

- This is what batch normalization does. 
- Although technically, we'll actually normalize the values of not `a[l-1]` but `z[l-1]` (some debates).


>  We might not want your hidden unit values be forced to have mean 0 and variance 1 in order to better take advantage of a non-linear activation function:
>> So in last step we trasform `z[l]_normed` to `z_tilde[l]`,  where (mean=β, std=gamma) and β and gamma are learnable params.
>> an that would exactly invert `z[l]_normed` equation.
---

# Fitting Batch Norm into a neural network

### Add batchnorm to NN: replace z[l] to z_tilde[l] at each layer before activation g[l].

<img src="https://upscfever.com/upsc-fever/en/data/deeplearning2/images/BATCH-NORMALIZATION-parameters.png" width="700">

### Extra params to learn at each layer.: 
- `gamma[l]` 
- `β[l]`
 

> - Dimension of beta[l], gamma[l]: the same as b[l] ( = n[l] * 1).
> - Batch Norm tries to learn is a different Beta than the hyperparameter Beta used in momentum and the Adam and RMSprop algorithms.
> - No bias term (b) in Batch Norm: → `b[l]` is replaced by `beta[l]`
>>  became `z[l] = W[l] * a[l-1]`


## Batch normalization with mini-batch Implementation:

#### Batch Norm does, is it is going to look at the mini-batch and normalize Z[L] to first of mean 0 and standard variance, and then a rescale by Beta and Gamma.

<img src="https://miro.medium.com/max/798/1*kyVa9UTnMIpOYUE0DDcb2A.png" width="500">

---
# Why does Batch Norm work?

## First Intuition : similar to normalizing input ("make contours round")

## Second intuition : weights in deeper layers are more robust to changes in ealier layer weights.
### i.e. Robost to data distribution changing. ("covariant shift")

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image016.png" width="600">

#### then you might not expect a module trained on the dataset on left to generalize well on the data on the right, even with the same ground truth function.
#### the idea is that, if you've learned some X to Y mapping, if the distribution of X changes, then you might need to retrain your learning algorithm.
#### With BN a[2] are ensured to always have the same mean/variance, so "data distribution" is unchanged → later layers can learn more easily, independent of previous layer's weights' change.

```
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
```

## BN as regularization

```
Since each minibatch is scaled by mean/var of just that minibatch,
adding noise to the transformation from z[l] to z_tilde[l], similar to dropout, add noise to each layer's activations.
therefor, BN have (slight) regularization effect (thie regularization effect gets smaller as minibatch size grows).
```
---
# Batch Norm at test time

### Batch norm processes your data one mini batch at a time, but the test time you might need to process the examples one at a time.

### at test time :
- → estimate the value of meu/sigma2
- → using exp-weighted-avg estimator across minibatchs (with beta close to 1 → ~running average).
- → just use the latest value of this exp-weighted-avg estimation as meu/sigma2.

<img src-"https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image019.png">

---

# Multiclass classification (Softmax Regression)

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image021.png">

## Activation function: softmax

<img src="https://miro.medium.com/max/1906/1*ReYpdIZ3ZSAPb2W8cJpkBg.jpeg" width="550">

> The sigmoid and the Relu activation functions input a real number and output a real number. However, the softmax activation function is unusual because it takes a vector instead of scalar anf then outputs of vector.

> The decision boundary between any two classes will be more linear.

---
# Training a classifier with a softmax output layer

### Softmax function generalizes the logistic activation function to C classes rather than just two classes.

# Loss function

## When the number of classes is 2, Binary Classification --> Just a logistic regression 

<img src="https://miro.medium.com/max/1050/1*-nJTj5mXtAAz9F107398sQ@2x.png" width="550">

## When the number of classes is more than 2, Multi-class Classification

<img src="https://miro.medium.com/max/1122/1*wcTVsYh3d4IQnYcUGdu66Q@2x.png" width="450">

`backprop:
dZ[L] = y_hat - y`
---

# Deep learning frameworks

<img src="https://x-wei.github.io/images/Ng_DLMooc_c2wk3/pasted_image029.png" width="550">

---

# TensorFlow

### Writing and running programs in TensorFlow has the following steps:

- Create Tensors (variables) that are not yet executed/evaluated.
- Write operations between those Tensors.
- Initialize your Tensors.
- Create a Session.
- Run the Session. This will run the operations you'd written above.
```

Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities,
but did not evaluate its value. To evaluate it, we had to run init=tf.global_variables_initializer().
That initialized the loss variable, and in the last line we were finally able to evaluate the value of loss and print its value.
```


- Define parameter to optimize:
```python
w = tf.Variable(0, dtype=tf.float32)
```

- define cost function: `cost = w**2 - 10 * w + 25`

- tells tf to minimize the cost with GD optimizer:
```python
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
```
- Create and run the session
```python
with tf.Session() as session:  
  session.run(init)  
  session.run(train)
```

- To inspect the value of a parameter: 
```python 
print(session.run(w))
```

- To Run 1000 iters of GD:
```python
for i in range(1000): session.run(train)
```

- Let loss function depends on training data:
    - define training data as placer holder.
    - a placerholder is a variable whose value will be assigned later.

```python
x = tf.placeholder(tf.float32, [3,1]) cost = x[0][0] * w**2 + x[1][0] * w + x[2][0]
```

- feed actual data value to placerholder: use `feed_dict` in session.run()
```python
data = np.array([1., -10., 25.]).reshape((3,1) session.run(train, feed_dict={x: data})
```

