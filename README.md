# Handwritten Digit Recognition CNN Model 
This model was built using a convolutional neural network (CNN) for classifying hand-written digits (0-9) from an image. 
This model was trained and tested on the Modified National Institute of Standards and Technology (MNIST) dataset. On the testing dataset, 
the model had an accuracy of 99.30%, which indicates an effective performance for recognizing hand-written digits. 

## Methodology
Our project used the tensorflow library from python to build the CNN because it is free, relatively easy to use, and it is a popular choice among many machine learning programs. In addition, we used the platform Google Collab to run our code because it provides a GPU, its collaborative nature, and it requires minimal set up. The methodology for this project was based on [3], [4]and [5], which also developed a CNN to recognize the MNIST handwritten digit dataset.

### Preprocessing the Dataset
Firstly, the training and testing dataset was loaded to the notebook from the MNIST dataset. This dataset contains 60,000 training images and 10,000 test images,
which consists of black and white digits, centered in the image, and made up of 28 x 28 pixels, and thus has a dimensionality of 784 (=28*28) [6]. Once the dataset 
was loaded in the notebook, the training and testing features were reshaped into 1D arrays. Then, the targets were one-hot-encoded to a binary class matrix, so that 
the current digit value would equal one, while the values for the rest of the digits would be zero. For example, if the digit was 1, then the output should look like
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]. Then, the training and testing features were converted to floats, and were normalized to range from 0 to 1 by dividing each value by 255.0.

### Model
A sequential model was built using Keras, which is an open-source library in python for developing neural networks. As shown in Fig.1, the model consists of a convolution layer 
with 32 filters, max pooling layer, convolution layer with 64 filters, max pooling layer, convolution layer with 120 filters, flatten layer, a dense layer with 120 nodes, and 
finally a dense layer mapped to the number of targets, which is 10, and used softmax as the activation function. All the convolution layers used same padding, a filter size of (3,3),
and ReLU as the activation function. The model was trained for 100 epochs, used categorical cross entropy as the loss function, and Adaptive Moment Estimation (Adam) as the optimizer
with a learning rate of 3e-4.
