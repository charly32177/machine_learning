# Machine Learning

Using three layers (two hidden layers and one output layer) of simple ANN with gradient descent.

## Dataset

The data contain gray-scale images of hand-drawn digits, from zero through nine from MNIST database.
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. In this project, we use ```42000``` data sample in total.

<img src="https://github.com/charly32177/machine_learning/blob/master/digits_predict.png" width="400">


We also normalize the values of the images into between 0 - 1.

<img src="https://github.com/charly32177/machine_learning/blob/master/digits_normalize.png" width="400">

---
## Code description


`nncost.py` define the cost function in `costfunction(...):` with following definition:
 
<img src="http://chart.googleapis.com/chart?cht=tx&chl= J(\theta) = \frac{1}{m}[\sum_{i=1}^m -y^{(i)} log( h_\theta(x^{(i)}) )-( 1- y^{(i)}) log( 1 - h_\theta( x^{(i)} ) ) ] %2B \frac{\lambda}{2m} \sum_{j=1}^{n_j}\sum_{i=1}^{n_i} \theta_{ij}^2  " style="border:none;">

The regulization parameter <img src="http://chart.googleapis.com/chart?cht=tx&chl= \lambda  " style="border:none;"> is obtained by using cross validation set:

![image](https://github.com/charly32177/machine_learning/blob/master/lambda_curve.png)

Value <img src="http://chart.googleapis.com/chart?cht=tx&chl= \lambda = 3.4  " style="border:none;"> is chosen.

`gradtheta(...):` define the derivative <img src="http://chart.googleapis.com/chart?cht=tx&chl= \frac{d J(\theta)}{d \theta_{ij}}  " style="border:none;"> with the `backpropagation` method, we also include 

`Numgradtheta(...):` function to do the gradient check with numerical method.


`nntrain.py` define the optimization method for gradient descent. We use [scipy.optimize.fmin_cg](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.fmin_cg.html) to iterate the weight 

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \theta_{ij} := \theta_{ij} - \alpha \frac{d J(\theta)}{d \theta_{ij}}    " style="border:none;">

---
## Run the code

For running the ANN machine learning, just
```
python machinelearning.py
```
The Neural Network architecture are
```
input_layer_size : 784
hidden_layer1_size : 50
hidden_layer2_size : 30
num_labels : 10
```

---
## Analysis

The `fmin_cg` terminted output shown below:
```
Optimization terminated successfully.
         Current function value: 0.164645
         Iterations: 3153
         Function evaluations: 8156
         Gradient evaluations: 8156
```

and also the plot of the cost function in each iter:
<img src="https://github.com/charly32177/machine_learning/blob/master/loss.png">

The training & validation set on recognition accuracy shown below:

```
train accuracy : 0.9998
vadiation accuracy : 0.9716
```
### Hidden unit

Visualization of Hidden Units:

<img src="https://github.com/charly32177/machine_learning/blob/master/hidden_units.png" width="400">

### Learning curve

The learning curve shows that this case is high variance case, It means that get more data can improve the accuracy.

<img src="https://github.com/charly32177/machine_learning/blob/master/learning_curve.png">

