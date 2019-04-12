## Numpy based Artificial Neural Network (Raw)

In this code we will implement an L-Layered Neural Network (ANN) and train it on MNIST dataset.
**Note** : We will not use any predefined libraries for this model building, this is raw neural network build only using numpy

**Author : Anugraha Sinha**

**Email : anugraha[dot]sinha[at]gmail[dot]com**

**Note** : 
Github's notebook view may stagnate. Use the link below for ease of viewage:

[numpy_based_ann_raw](https://nbviewer.jupyter.org/github/anugrahasinha/neuralnetwork/blob/master/numpy_based_ann_raw/numpy_based_ann_raw.ipynb)

### Background
There are many libraries available for building neural networks and its derivatives (like CNN, RNN, LSTM, GRU etc.). However, the core concept (forward propogation and backward propagation) is generally hidden behind the implementation of these libraries. 
It is important to understand the background mathematics which leads to results we see from neural networks. This code tries to show the background calculations involved in neural network (Simple ANN - Artificial Neural Networks).

### Motivation
The idea behind this implementation is to build a raw neural network and showcase how mathematics work behind the "black box" of neural networks. Normally, it is intriguing and information to understand the theory behind the working of neural networks. However, implementation of complex mathematical function becomes tiresome for proving the mathematical concept from a programming perspective. This implementation is aimed at bridging that gap.

#### Forward Propagation
Forward propagation is fairly simple operation, which involves 
- Utilizing activation function at respective hidden layers / Output layers
- matrix multiplication and bias additions

However, the code learning behind neural network lies with back propagation of the gradient/loss calculated with each forward propagation operation.

#### Backward Propagation
The core learning concept behind neural network is as to how the weights and biases are modified so that loss at the output layer can be reduced.
This involves differentiation of loss function with respect to individual weight/biases to adjust them based on the learning rate.


#### Current implementation

- The current implementation employs
  - ReLU based activation function for hidden layers
  - Softmax based output layers
  - Loss function being entropy loss
  - works with multi-class classification problems (MNIST dataset shown in the code)
  - Uses Gradient Descend fundamental for changing weights with each iteration
  
- What is not shown
  - Regularization implementation
  - Batch Normalizations
  - Advance optimization technique/algorithms like Adam Optimization.
  
#### Results
Using MNIST dataset (subset of it - only 5000 images as training data) we can see that network is able to understand the inherent information hidden in the pixels of images and build a fairly acceptable model with close to 88% (training accuracy) and 87% (test accuracy). The code is robust enough to be trained further if computing resources and time is available.