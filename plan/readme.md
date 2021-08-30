# Intern Project Plan

## Company
Shenzhen DeepVision Innovation Technology Co., Ltd

http://www.deepvai.com/en/index.html


https://github.com/bentrevett/pytorch-image-classification

https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864

https://www.youtube.com/watch?v=aircAruvnKk
## 8 Weeks Plan
* 2 weeks to learn related skills
* 4 weeks to do real work
* 2 weeks to wrap up

## 1st week
* Learn GIT. Follow tutorial to learn GIT, for example, https://www.codecademy.com/learn/learn-git
* Learn Command line. Follow tutorial to learn Windows Command Line, such as https://www.codecademy.com/learn/learn-the-command-line
* Learn Markdown. Follow tutorial to learn Markdown, such as https://www.markdowntutorial.com/ and https://www.markdownguide.org/basic-syntax/. All of the documentatio will be done using markdown.
* Learn advanced Python. Read book "Learn Python 3 the Hard Way" and "Learn More Python 3 the Hard Way"

## 2nd week
* Continue learn advanced Python knowledge
* Watch https://www.youtube.com/watch?v=aircAruvnKk
* Watch https://www.youtube.com/watch?v=IHZwWFHWa-w&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=2
* Watch https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=3
* Watch https://www.youtube.com/watch?v=tIeHLnjs5U8&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=4
* Start to learn tutorial in https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864


* Start to learn tutorial in https://github.com/bentrevett/pytorch-image-classification
* Tutorial https://towardsdatascience.com/neural-networks-for-beginners-by-beginners-6bfc002e13a2

## 3rd week

### Learn NumPy
If you get error that numpy is not defined, please run
```
pip install numpy
```
Try the following code to understand NumPy library. You could also try to write your own code to use NumPy library
```python

import numpy as np

print(np.sum([1,2,3]))

matrixA = [[1,2,3], [4,5,6]]
vectorA = [1,2,3]
print(np.dot(matrixA, vectorA))
matrixB = [[1,2], [3,4], [5,6]]
print(np.dot(matrixA, matrixB))

matrixC = [[1,2,3], [1,2,3]]
print(np.subtract(matrixA, matrixC))

print(np.add(matrixA, matrixC))

print(np.add(matrixA, 1))

print(matrixA)

print(np.array(matrixA))

```

Create your own class Matrix, which has 
```python
class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for j in range(cols)] for i in range(rows)]
```

and add a few methods on it, so that you could perform dot, add, substruct operations. You will define static methods, for example
```python
    @staticmethod
    def add(matrixA, matrixB):
        if (matrixA.rows != matrixB.rows):
            raise Exception("matrix rows not match")
        if (matrixA.cols != matrixB.cols):
            raise Exception("matrix cols not match")
        
        ret = Matrix(matrixA.rows, matrixA.cols)
        for i in range(matrixA.rows):
            for j in range(matrixA.cols):
                ret.data[i][j] = matrixA.data[i][j] + matrixB.data[i][j]
        return ret
```

And then you could compare your Matrix implementation with numpy implementation.

### Perceptron
- Learn the simplest nenural network, Perceptron, algorithm and implement it using pure Python
https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

https://www.youtube.com/watch?v=ntKn5TPHHAk&list=RDCMUCvjgXvBlbQiydffZU7m1_aw&index=4

A sample perceptron is:
```python
import random

class Perceptron:
    def __init__(self, inputNodeCount):
        # initialize the weight with random number from -1 to 1
        self.weights = [random.uniform(-1, 1) for x in range(inputNodeCount)]
        # initialize bias
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.02

    def guess(self, inputs):
        sum = self.bias
        for i in range(len(self.weights)):
            sum += self.weights[i] * inputs[i]

        # use very simple function as activation function
        # It only return -1 or 1 based on sign
        if (sum < 0):
            return -1
        return 1

    def train(self, inputsAndTargetList):
        sum_error = 0
        for inputsAndTarget in inputsAndTargetList:
            inputs = inputsAndTarget[0: len(self.weights)]
            # last element is the target
            target = inputsAndTarget[-1]
            prediction = self.guess(inputs)
            err = target - prediction
            sum_error += err* err
            self.bias = self.bias + self.learning_rate * err
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] + self.learning_rate * err * inputs[i]
        print(f'error={sum_error}')


class Point:
    def __init__(self, x = None, y = None):
        if (x is None):
            x = random.uniform(0, 300)
        if y is None:
            y = random.uniform(0, 300)

        self.x = x
        self.y = y
        if x > y:
            self.label = 1
        else:
            self.label = -1

def train_and_predict():
    trainingPoints = [Point() for i in range(100)]
    inputsAndTargetList = [[p.x, p.y, p.label] for p in trainingPoints]

    # create a brain, who only has 2 inputs
    brain = Perceptron(2)
    for i in range(5):
        brain.train(inputsAndTargetList)

    print(f"bias={brain.bias}, weights={brain.weights}")

    testingPoints = [Point() for i in range(50)]
    correctCount = 0
    for pt in testingPoints:
        prediction = brain.guess([pt.x, pt.y])
        if (prediction != pt.label):
            print(f"Wrong prediction for ({pt.x}, {pt.y})")
        else:
            correctCount+= 1
    # print out accuracy
    print(f"{correctCount} out of {len(testingPoints)} are correct")

if __name__ == "__main__":
    train_and_predict()


```

### How weight/bias is updated
In the above code, for each training cycle, we will update the weight and bias to new value.
```
w0_new = w0 + learning_rate * error * x0
bias_new = bias + learning_rate * error
```
You may wonder why. It's all because we want to minimize the square of the error. Because error could be positive or negate, we use square of the error. If the square of the error is minimized, we get the appropriate weight and bias.

Suppose we have
```
prediction = w0 * x0 + w1 * x1 + w2 * x2 + b
```
We will know that
```
derivative_of_prediction_over_w0 = x0
derivative_of_prediction_over_w1 = x1
derivative_of_prediction_over_b = 0
...
```

If the `target` is the actual answer. Then we know the error is
```
error = target - prediction
error_square = error * error = (target - prediction) * (target - prediction)

derivative_of_errorSquare_over_w0 = 2 * (target-prediction) * (-1) * derivative_of_prediction_over_w0
   = 2* error * (-1) * x0
```

We could then apply the gradient descent method. If the `derivative_of_errorSquare_over_w0` is positive, we should decrease w0 a little bit. If `derivative_of_errorSquare_over_w0` is negative, we should increase w0 a little bit. Let the `step_size` to be a very small number, such as 0.01, we will have

```
w0_new = w0 - step_size * derivative_of_errorSquare_over_w0

w0_new = w0 + step_size * 2 * error * x0
```

If we define
```
learning_rate = step_size * 2
```
We will have
```
w0_new = w0 + learning_rate * error * x0
```

That's why when we try to find the new weight, we will use the older weight, then plus learning_rate times error and times the input. You could do similar exercise to get the formula to update the bias.

### Understand Neural Network
- Continue to watch the "The Coding Train" series of "Neural Netowork" to understand Feed Forward and Back Propagation. Please watch 
1. Multiple layer perceptron: 10.4, 10.5
2. Feed Forward: 10.12, 10.13
3. Back Propagation: 10.14, 10.15, 10.16, 10.17, 10.18

In YouTube.com, you could type in "neural network coding train 10.4"

- Also watch https://www.youtube.com/watch?v=w8yWXqWQYmU "Building a neural network FROM SCRATCH", which will help you to understand the Neural Network

- Follow the "Building a neural network from scratch" Youtube video and start to rebuild the project.
- The detailed blog is https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy.html

## 4th week
- Finish the project by follow the https://www.youtube.com/watch?v=w8yWXqWQYmU "Building a neural network FROM SCRATCH"

- A reference is at https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy.html

Please note that there is an error in the formula used by the above YouTube.com video and the blog as it did not have correct diverative_of_loss_over_w2. The correct formula is in Samson's jupyter notebook. https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras. When we do implementation, please use the correct formula from Samson's jupyter notebook.

- Also watch [Neural Networks Explained from Scratch using Python](https://www.youtube.com/watch?v=9RN2Wr8xvro)

### MNIST dataset
NIST (National Institute of Standard Technology) collected hand written from a lot of people and put the image into a CSV (Comma Separated Value) file. Then someone modified the data and called it MNIST (Modified NIST) dataset. You could download it from https://www.kaggle.com/oddrationale/mnist-in-csv

The first row is the coloumn header, the rest rows has 28x28 + 1 = 784 + 1 = 785 columns. The first column is the integer lable for the digit. The other 784 columns are the gray scales from 0 to 255.

Please write code to read data from the file and get familar with the data. The following code requires numpy and matplotlib packages. It uses matplotlib to show visual image. If you get error when the package is not there, please
```
pip install numpy
pip install matplotlib
```

Sample code could be:
```python
from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):
    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        for row in csv_reader:
            if row is None:
                continue
            ret.append(row)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break

        return ret

def load_dataset(filename, max_rows):
    # as we will skip the first row, we need to load one more row from the CSV file
    if max_rows > 0:
        max_rows += 1
    csv_data = load_csv(filename, max_rows)

    # skip the first row as it just has column headers
    csv_data = csv_data[1:]

    # Create empty list
    dataset = list()
    labels = list()
    for raw_row in csv_data:
        # call int(s) to convert a string to int
        label = int(raw_row[0])
        labels.append(label)

        # for the column index from 1 to 784, convert the string to integer,
        # then divide it by 255 to get float number represent the scale of the gray in range 0 to 1
        row = [int(col) / 255.0 for col in raw_row[1:]]
        dataset.append(row)
    return dataset, labels

# show image for the data at index
def show_image(dataset, labels, index):
    label = labels[index]

    # reshape data to 28 x 28 matrix
    image = np.array(dataset[index]).reshape((28, 28))
    # convert 0 to 1 scale to 0 to 255
    image = image * 255
    print(f'label={label}')

    # now we want to plot a gray image in memory
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    # finally show it on the screen
    plt.show()

if __name__ == "__main__":
    # As there are a lot of rows, we only want to try to read small number of rows, such as 10
    dataset, labels = load_dataset('mnist_train.csv', 10)
    # choose a index. You could choose another value
    index = 8
    print(labels[index])
    print(dataset[index])
    show_image(dataset, labels, index)
```

### Neural Network using MNIST dataset
The YouTube video [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU) and Samson's [blog](https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy.html) explained how to build Neural Netwrok using MNIST dataset. However, there is an error in the formula used by the above YouTube.com video and the blog as it did not have correct diverative_of_loss_over_w2. The correct formula is in Samson's [jupyter notebook](https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras). When we do implementation, please use the correct formula from Samson's jupyter notebook.

#### NumPy Tricks
Before you implement the NeuralNetwork, please get more familar with NumPy. Especially some tricks that people will easily get unexpected results.
```python
# Please pay attention when we mix Matrix and Vector operation using numpy as there could be unexpected result
import numpy as np
m1 = np.array([[1,2,3],[4,5,6]])
v1 = np.array([1,2,3])
r1 = m1.dot(v1)
print(r1)

v2 = np.array([10,12])
print(r1 + v2)

r2 = m1.dot(v1.reshape(-1,1))
print(r2)

# (r2 + v2) is supposed to be same as (r1 + v2), howerver, they are different 
print(r2 + v2)
```

To avoid unexpected results, it's better to always convert vector to matrix for the operations. For example, we could use
```python
# convert list [1,2,3] to a matrix with one column.
np.array([1,2,3]).reshape(-1,1)
```

#### Neural Network
We will build the Neural Network on top the code that read MNIST.

```python
from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):
    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        for row in csv_reader:
            if row is None:
                continue
            ret.append(row)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break

        return ret

def load_dataset(filename, max_rows):
    # as we will skip the first row, we need to load one more row from the CSV file
    if max_rows > 0:
        max_rows += 1
    csv_data = load_csv(filename, max_rows)

    # skip the first row as it just has column headers
    csv_data = csv_data[1:]

    # Create empty list
    dataset = list()
    labels = list()
    for raw_row in csv_data:
        # call int(s) to convert a string to int
        label = int(raw_row[0])
        labels.append(label)

        # for the column index from 1 to 784, convert the string to integer,
        # then divide it by 255 to get float number represent the scale of the gray in range 0 to 1
        row = [int(col) / 255.0 for col in raw_row[1:]]
        dataset.append(row)
    return dataset, labels

# show image for the data at index
def show_image(dataset, labels, index):
    label = labels[index]

    # reshape data to 28 x 28 matrix
    image = np.array(dataset[index]).reshape((28, 28))
    # convert 0 to 1 scale to 0 to 255
    image = image * 255
    print(f'label={label}')

    # now we want to plot a gray image in memory
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    # finally show it on the screen
    plt.show()

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum


class NeuralNetwork:
    def __init__(self, inputCount, hiddenCount, outputCount):
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        
        # initialize
        self.W1 = np.random.rand(hiddenCount, inputCount) - 0.5
        self.b1 = np.random.rand(hiddenCount, 1) - 0.5
        self.W2 = np.random.rand(outputCount, hiddenCount) - 0.5
        self.b2 = np.random.rand(outputCount, 1) - 0.5
        self.learning_rate = 0.01
    
    def feed_forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    
    def one_hot(self, Y):
        ret = [0 for x in range(self.outputCount)]
        ret[Y] = 1
        # return matrix with 1 column
        return np.array(ret).reshape(-1,1)
    
    def backward_prop(self, X, Z1, A1, Z2, A2, Y):
        one_hot_Y = self.one_hot(Y)

        dZ2 = A2 - one_hot_Y
        dW2 = dZ2.dot(A1.T)
        db2 = dZ2
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = dZ1

        # update the weights and bias
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2


    # train the network with inputs X and the expected answer Y
    def train(self, X, Y):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        self.backward_prop(X, Z1, A1, Z2, A2, Y)

    # predict the answer for the input
    def predict(self, X):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        # A2 is a matrix with 1 column, find the index with max value for axis 0
        index = np.argmax(A2, 0)
        return index


def train_and_predict():
    # As there are a lot of rows, we only want to try to read small number of rows, such as 10
    train_count = 5000
    test_count = 1000
    dataset, labels = load_dataset('mnist_train.csv', train_count + test_count)

    brain = NeuralNetwork(28 * 28, 10, 10)
    for i in range(train_count):
        # input is a matrix with 1 column
        X = np.array(dataset[i]).reshape(-1,1)
        # Y is expected value
        Y = labels[i]
        brain.train(X, Y)
    
    correct_count = 0
    for i in range(train_count, train_count + test_count):
        # input is a matrix with 1 column
        X = np.array(dataset[i]).reshape(-1,1)
        Y = labels[i]
        prediction = brain.predict(X)
        if prediction == Y:
            correct_count += 1
        else:
            print(f'incorrect prediction for data at index={i}. Prediction={prediction}, Answer={Y}')
    
    print(f'{correct_count} of {test_count} are correct')

if __name__ == "__main__":
    train_and_predict()
```

The core concept is `feed_forward` and `backward_propagation`. The activation function determine the derivative, which is used to adjust the weigts and bias.

You may wondering why `ReLU_deriv` is defined as
```python
def ReLU_deriv(Z):
    return Z > 0
```
It will return a matrix or vector with boolean value instead of number value. You could write a more easy-to-understand way to do it. But Python developers always mix the boolean values with numbers. The `True` is 1 and `False` is 0. Derivative of ReLU is 0 if x is less than 0, otherwise, it's 1. A more easy-to-understand way is something like

```python
def ReLU_deriv(Z):
    ret = list()
    for row in Z:
        new_row = list()
        for col in row:
            new_col = 1 if col > 0 else 0
            new_row.append(new_col)
        ret.append(new_row)
    return ret
```

The Python developer write it in `Z > 0` way only because Python treat `True` as 1 and `False` as 0 in math operations.

The above code will report some pictures that it did not predict correctly. You could write another program to display those pictures so that you could check it. For example
```python
# assume that the above code is in the mnistwork2.py, you could import it
import mnistwork2

dataset, labels = mnistwork2.load_dataset('mnist_train.csv', 10000)

while True:
    index_str = input("Type Index:")
    if index_str is None or len(index_str) == 0:
        break
    index = int(index_str)
    mnistwork2.show_image(dataset, labels, index)

```

By this way, you could manually check those images that your code did not predict correctly. You will see that some images are very tricky.

#### Correct Formula
The Correct Formula is in the [jupyter notebook](https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras). It's ![here](mnist-formula.png).


## 5th Week
### PyTorch
Right now, we understand the principal of Neural Network, which is to use gradient decent to find the minimal cost/loss function. We use feed_forward to get the cost/loss value, then apply the derivative formula to update the weights and bias.

For a three-layer Neural Network, it's easy to do it from scratch. We also see that the most difficult part in the three-layer Neural Network is the backward_propagation part. When we need to build more than three layers, the backward_propagation part is more difficult. The PyTorch library will help us to calculate the backward_propagation.

We will follow the tutorial published by "Python Engineer" to learn the basics in PyTorch library. Once we learn the basics, it will be easy to apply the knowledges on the MNIST dataset. The goal of this week is to be able to use PyTorch library to redo the MNIST dataset problem, but use multiple layers instead of just three layers.

There are 16 videos in the PyTorch Turotail. https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4  . You could skip the first one as the first one just talk about the installation. You could start with the second one, which talk about Tensor.

Notes:
- You could speed up using 1.5 as Playback speed while you watch the Youtube video.
- You could just watch the 2 to 13 and skip the 14,15,16 for now. You could watch No.14, 15, 16 later.
- In the turorial 05, you may not understand the sample code using pure NumPy library. That's OK. But you should try to run the code using Torch, as that's the fundamental of the PyTorch library. The following is the code for tutorial 05.

```python
import torch
import numpy as np

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return W * x

# loss = MSE (mean of error square)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    # let the PyTorch library to calculate the derivative_of_loss_over_weight
    l.backward()
    
    # update weights. As the PyTorch should not keep track of the gradient
    # on the weight update operation, we will use torch.no_grade()
    with torch.no_grad():
        W -= learning_rate * W.grad

    # After each training, the grad on the weight should be reset to zero
    W.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch {epoch}: W={W}, loss={l}")
```

- You could skip the PyTorch Tutorial 08 - Logistic Regression
- The PyTorch Tutorial 13 - Feed-Forward Neural Network is very important. The following is the code that uses MNIST dataset

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Inspect the data
print(len(train_dataset))
index = 4
img, label = train_dataset[index]
print(label)
print(img)
print(img.shape)
print(img.size())

# show images
# as the img shape is [1,28,28], we will get img[0], which is the data
# plt.imshow(img[0], cmap="grey")
# plt.show()


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Inspect loader
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch {epoch}, step {i} / {n_total_steps}, loss={loss}")


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuancy={acc}')

```

### Real DataSet
Now, we will use the dataset used in production. Please download the dataset from
https://drive.google.com/file/d/1KVBfMk5IBij2u2JGyE-WKG6WkcX_oqOA/view?usp=sharing

It's a ZIP file and you could unzip it to your own folder.

Let's explore the dataset a little bit to understand what are them. For example, there is `train_image_label` folder, which contains sub folders `label` and `srcImg`. Under those folders, are the images in bmp format.

Python has a lot of libraries that will help us to read the images. For example, we could use `os.listdir()` API to get the file names. Once we have a particular file name, we could use `os.path.join()` API to get the full path fo the file. Then we could use `PIL.Image.open()` to open the image in memory. Once it's loaded in memory, we could use `image.show()` to display it to make sure we got the correct code. Once we have the image, we will convert it to matrix data. PyTorch has `torchvision.transforms.functional.to_tensor()` API that will transform image to matrix data. Please note that the tensor data is three dimentional array instead of two dimensional array, which is similar as the MNIST dataset.

Please get familar with those Python APIs. The following are some sample code.


```python
import os
from PIL import Image
import torchvision.transforms.functional as TF

train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"

# List all of the files under the folder
src_image_names = os.listdir(train_src_image_folder)
print(src_image_names)
print(len(src_image_names))

# let's check one of the image
src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")


# open image
src_image = Image.open(src_image_full_path)
label_image = Image.open(label_image_full_path)

# show the image
print("src_image")
src_image.show()
print("label_image")
label_image.show()

# now convert it to Tensor
src_image_tensor = TF.to_tensor(src_image)
label_image_tensor = TF.to_tensor(label_image)

print(src_image_tensor)
# We will see that it's three dimension data
print(src_image_tensor.size())

# we only need to get the first element, which is two-dimension array.
print(src_image_tensor[0])

```

## 6th Week
### Tensor
Please note that tensor not only store values, but also keep track of the derivatives.
```python
import torch

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y = x*x + 3 x
# the derivative should be (2 x + 3)
b = a * a + 3 * a
loss = b.sum()
loss.backward()

# now could print the value
print(a)
# and the derivative
print(a.grad)


```
### Convolutional Neural Network
The real project requires understanding of Convolutional Neural Network. 
1. Please watch the following video to understand what is "Filter", "Stride" and "Max Pooling". As the code in the following two vides are written in JavaScript, you do **not** need to write code, but try to understand the concept.

- https://www.youtube.com/watch?v=qPKsVAI_W6M&t=940s
- https://www.youtube.com/watch?v=pRWq_mtuppU

2. Then read the article that explain the concepts. https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29 . As you may not have time, you do **not** need to try the code in this article, but please try to understand the concept of "channels", "kernel" or "filter", "stride" and "max pooling"

3. Watch video "PyTorch Tutorial 14 - Convolutional Neural Network (CNN)", https://www.youtube.com/watch?v=pDdP0TFzsoQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=14. As you may not have time, you do **not** need to try the code there.

4. We will also use `torch.save()`, `torch.load()`, `model.dict_state()` API to save the model. Those API are explained in the "PyTorch Tutorial 17 - Saving and Loading Models", https://www.youtube.com/watch?v=9L9jEOwRrCg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17 . Please watch the video to understand how the API will be used.

5. There is cs231n course from Stanford. The course is listed at YouTube. https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk. While it may be too much to watch, it's very useful and it talks about the details of convolutional neural network.

### Semantic Segmentation
For this project, we will build a model based on the existing FCN ResNet or DeepLab model to detect the defect.

Please read the article 
https://towardsdatascience.com/semantic-hand-segmentation-using-pytorch-3e7a0a0386fa
to understand how to build your own model based on the existing model built in PyTorch. When you read the article, you do **not** need to write the code, but trying to understand the concept.

As we know, it will be difficult for everyone to rebuild some advanced model. That's why PyTorch provided some library for us to use them directly. We could build our own model based on the exsting model included int the PyTorch library. For example
```python
class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret
```

### Some basic PyTorch and Image APIs
We will use a few more PyTorch and Image APIs. Please get familar with those APIs

#### squeeze/unsqueeze
```python
import torch

# one dimensional array
a = torch.tensor([1,2,3])
print(a)
print(a.shape)

# convert to two dimensional array
b = torch.unsqueeze(a, dim=0)
print(b)
print(b.shape)

# convert to three dimensional array
c = torch.unsqueeze(b, dim=0)
print(c)
print(c.shape)

# convert back to two dimensional array
d = c.squeeze(dim = 0)
print(d)
print(d.shape)

# the squeeze could cause data loss
e = torch.tensor([[10,20,30],[40,50,60]])
f = c.squeeze(dim = 0)
print(f)
print(f.shape)
```


#### Image API
```python
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.


current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")

# open image
src_image = Image.open(src_image_full_path)
print(src_image.size)
src_image_rgb = src_image.convert("RGB")
src_image_rgb.show()

label_image = Image.open(label_image_full_path)
print(label_image.size)

# get image data into np.array
src_image_data = np.array(src_image, dtype=np.float32)
print(f"src_image_data.shape={src_image_data.shape}")
print(src_image_data.max())
# convert to 0 to 1
src_image_data = src_image_data / 255.0

label_image_data = np.array(label_image, dtype=np.float32)
print(label_image_data.max())
# replace all element greater than 0 with value 1.0
# we only want to use 0 or 1
label_image_data[label_image_data > 0] = 1.0
print(label_image_data.max())

# show two images
fig, axarr = plt.subplots(2)
# show the first image with src_image_data
axarr[0].imshow(src_image_data, cmap="gray")
# show the second image with label_image_data
axarr[1].imshow(label_image_data, cmap="gray")
plt.show()


# When we pass the value to Full Convolutional Network, the input should have three channel, we will also resize it.
# please note that we use // operator so that we will get integer value
image_width = 716 // 4
image_height = 920 // 4
src_image = Image.open(src_image_full_path).convert("RGB")
preprocess = transforms.Compose([
                            transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
src_image_data = preprocess(src_image)
print(src_image_data.shape)

# the above code using transforms.Compose() has the same result as
y = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)(src_image)
y = transforms.ToTensor()(y)
y = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(y)

# now, process the label image
label_image = Image.open(label_image_full_path).convert("L")
# create a function that will be used to resize the image
trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
label_image_resized = trf_resize(label_image)
# create numpy data from the image
label_image_data = np.array(label_image_resized)
print(label_image_data)
# we will then create two classes, the first is the background and the second is the label.
defect_class = np.zeros_like(label_image_data)
defect_class[label_image_data > 0] = 1.0
# background class
background_class = np.zeros_like(label_image_data)
background_class[label_image_data == 0] = 1.0
label_image_classes = torch.tensor([background_class, defect_class])
print(label_image_classes.shape)


```

#### Conv2d
This is to show the effects of Conv2d

```python
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"
src_image = Image.open(os.path.join(train_src_image_folder, src_image_name))

src_image_data = np.array(src_image, dtype=np.float32)

src_image_tensor = torch.tensor(src_image_data)
print(src_image_tensor.shape)

# The data is 2 dimensional, convert it to 4 dimensional. It's because nn.Conv2d requires 4 dimensional data. 
# the first dimensional is the the index in the batch
# the second dimensional is channel
# the third and fourth is the 2D data
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
print(src_image_tensor.shape)


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_result = conv.forward(src_image_tensor)
print(conv_result.shape)

conv_result_data = conv_result.squeeze().detach().numpy()

fig, axarr = plt.subplots(2)
# the first is source image
axarr[0].imshow(src_image_data, cmap="gray")
# the second is the image after convolutional process
axarr[1].imshow(conv_result_data, cmap="gray")
plt.show()

```

### Segmentation
We will build your own model based on the existing model included in the PyTorch library, then train it with our own data and test it. The following is the sample code.

Please note that it may take very long to train the model. You could update the `num_epochs` to 1, or even reduce the `image_width` and `image_height` further.

You also could try to replace the `deeplabv3_resnet50` with `fcn_resnet50` to show the impact.

You also could try to replace the loss function with `MSELoss` to show the impact.


```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import os
import matplotlib.pyplot as plt


# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")

batch_size = 2
# make the size smaller so that we could run it quickly
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.0001
num_epochs = 4

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.src_img_names = os.listdir(src_image_folder)
        mod = len(self.src_img_names) % batch_size
        # remove the entries that not fall into the batch
        self.src_img_names = self.src_img_names[0: len(self.src_img_names) - mod]
        print(len(self.src_img_names))


    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc).convert("RGB")

        preprocess = transforms.Compose([
                                    transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc).convert("L")
        label_image_resized = trf_resize(label_image)
        # create numpy data from the image
        label_image_data = np.array(label_image_resized)
        # we will then create two classes, the first is the background and the second is the label.
        defect_class = np.zeros_like(label_image_data)
        defect_class[label_image_data > 0] = 1.0
        # background class
        background_class = np.zeros_like(label_image_data)
        background_class[label_image_data == 0] = 1.0
        label_image_classes = torch.tensor([background_class, defect_class])

        return src_image_data, label_image_classes


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret



def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)

    fig, axarr= plt.subplots(2,3)

    # the image has three channels
    axarr[0,0].imshow(src_image[0], cmap = "gray")
    axarr[0,1].imshow(src_image[1], cmap = "gray")
    axarr[0,2].imshow(src_image[2], cmap = 'gray')

    # the labels has two, the first element is background, the second element is the label
    axarr[1,0].imshow(label_image[0], cmap="gray")
    axarr[1,1].imshow(label_image[1], cmap="gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SegmentationModel()

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    outputs = model.forward(inputs)
    print(outputs.shape)
    print(labels.shape)

    targets = labels.float()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    print(loss)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loop(train_loader, model, criterion, optimizer)

def load_checkpoint_and_train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    checkpoint = torch.load("saved-deeplab-checkpoint.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loop(train_loader, model, criterion, optimizer)


def train_loop(train_loader, model, criterion, optimizer):
    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            predicts = model(images)

            # loss
            targets = labels.float()
            loss = criterion(predicts, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #   break

    torch.save(model, os.path.join(current_location, "saved-deeplab.pth"))
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, os.path.join(current_location, "saved-deeplab-checkpoint.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab.pth"))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        # only show the one that we are interesting
        # 20, 30 are OK, but
        # 40, 42 are not OK.
        target_index = 45
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)
                target = label.float()
                loss = criterion(output, target)
                print(f"loss={loss}")

                output_raw = output.squeeze(0)

                output_background = output_raw[0]
                output_defect = output_raw[1]

                background_img = transforms.ToPILImage()(output_background)
                defect_image = transforms.ToPILImage()(output_defect)
                # we could show the image
                # background_img.show()
                # defect_image.show()

                # the index who has the max value. As we only have two classes, it's either 0 or 1
                output_label = output_raw.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # if the defect class is larger than background, it's defect.
                # ytest should be same as output_label
                ytest = output_raw[1] > output_raw[0]

                # draw the images
                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1]

                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                axarr[0,1].imshow(torch.sigmoid(output_raw[0]), cmap="gray")

                # defect
                axarr[0,2].imshow(torch.sigmoid(output_raw[1]), cmap="gray")

                # expected label
                axarr[1,0].imshow(label_data, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                # show ytest, it should be same as output_label
                # axarr[1,2].imshow(ytest, cmap="gray")
                plt.show()


            index += 1

def test_predict_using_pretrained_standard_model():
    """
    Predict the object using pretrained standard model
    We will see that the standard model will not work for us
    """
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
    model.eval()
    with torch.no_grad():
        target_index = 20
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)["out"]
                target = label.float()

                output_classes = output.squeeze(0)

                # the index who has the max value
                output_label = output_classes.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # draw the images
                f, axarr = plt.subplots(2,2)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1]

                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                # axarr[0,1].imshow(torch.sigmoid(output_classes[0]), cmap="gray")

                # expected label
                axarr[1,0].imshow(label_data, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                plt.show()


            index += 1


if __name__ == "__main__":
    # test_imagedataset()
    # test_train_one()
    # train_and_save_model()
    # load_checkpoint_and_train_and_save_model()
    test_model()
    # test_predict_using_pretrained_standard_model()

```

## Extra credit
For the extra credit, we will implement the U-NET

1. Watch video "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby", https://www.youtube.com/watch?v=IHq1t7NxS8k, this is the **most important** one and we will follow the tutorial to finish the project. The author also published the code at the github https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet. We **do** need to understand most of the core code.


1. To understand U-Net algorithm, you could watch "U-NET Paper Walkthrough", https://www.youtube.com/watch?v=oLvmLJkmXuc


For the project, we will follow the tutorial "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby", but modify it based on our training data. We could also simplify it a little bit.


### Defect Detection using UNET
The code is adapted from "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby", https://www.youtube.com/watch?v=IHq1t7NxS8k. We provide our own Dataset to read the data. We also tried to simplify the code. Please follow the YouTube video to understand how U-NET works. Please also check the author's code at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

Please modify the code inside the main. When we train the model, it will save the model. When we test the model, we load the model from the file.

```python
if __name__ == "__main__":
    # test_UNET()
    # test_imagedataset()
    # test_loader()
    train_model()
    # test_model()
```

The following is the full code.

```python
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")

batch_size = 2
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.001
num_epochs = 2

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder, transform = None):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.transform = transform
        self.src_img_names = os.listdir(src_image_folder)

    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc)
        # src_image = src_image.resize((image_width, image_height), Image.NEAREST)
        src_image_data = np.array(src_image, dtype=np.float32)
        src_image_data = src_image_data / 255.0
        # src_image_data = torch.tensor(src_image_data, dtype=torch.float32)


        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc)
        # label_image = label_image.resize((image_width, image_height), Image.NEAREST)
        label_image_data = np.array(label_image, dtype=np.float32)
        label_image_data[label_image_data > 0] = 1.0
        # label_image_data = torch.tensor(label_image_data, dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(image=src_image_data, mask=label_image_data)
            src_image_data = augmentations["image"]
            label_image_data = augmentations["mask"]

        return src_image_data, torch.unsqueeze(label_image_data, dim=0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[:: -1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

image_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            ToTensorV2(),
        ],
    )


def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)
    plt.imshow(src_image.squeeze(0), cmap = "gray")
    plt.show()
    plt.imshow(label_image.squeeze(0), cmap="gray")
    plt.show()

def test_UNET():
    x = torch.randn((2, 1, 920, 716))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)

    x = torch.randn((1, 920, 716))
    x = torch.unsqueeze(x, dim=0)
    preds = model(x)
    print(preds.shape)

def test_loader():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    examples = iter(train_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)


def train_model():

    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = UNET(in_channels=1, out_channels=1)


    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #     break

    torch.save(model.state_dict(), os.path.join(current_location, "saved.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder, image_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = UNET(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(current_location, "saved.pth")))
    model.eval()
    with torch.no_grad():
        target_index = 20
        index = 0
        for images, labels in test_loader:
            if index == target_index:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                print(outputs.shape)
                print(outputs)

                f, axarr = plt.subplots(2,2)
                axarr[0,0].imshow(images.squeeze(0).squeeze(0), cmap = "gray")
                axarr[1,0].imshow(labels.squeeze(0).squeeze(0), cmap = "gray")
                axarr[1,1].imshow(outputs.squeeze(0).squeeze(0), cmap = "gray")
                plt.show()


            index += 1




if __name__ == "__main__":
    # test_UNET()
    # test_imagedataset()
    # test_loader()
    train_model()
    # test_model()

```

During the training, you may notice that the loss get smaller and smaller, but has fluctuation.
```text
epoch 0, step 0 / 358, loss=0.7198129892349243
epoch 0, step 20 / 358, loss=0.28176361322402954
epoch 0, step 40 / 358, loss=0.18625952303409576
epoch 0, step 60 / 358, loss=0.12518161535263062
epoch 0, step 80 / 358, loss=0.09668765962123871
epoch 0, step 100 / 358, loss=0.08442027866840363
epoch 0, step 120 / 358, loss=0.06081239879131317
epoch 0, step 140 / 358, loss=0.0702969953417778
epoch 0, step 160 / 358, loss=0.04157618060708046
epoch 0, step 180 / 358, loss=0.034358784556388855
epoch 0, step 200 / 358, loss=0.040915440768003464
epoch 0, step 220 / 358, loss=0.027041830122470856
epoch 0, step 240 / 358, loss=0.018930351361632347
epoch 0, step 260 / 358, loss=0.02023766189813614
epoch 0, step 280 / 358, loss=0.021855659782886505
epoch 0, step 300 / 358, loss=0.027157608419656754
epoch 0, step 320 / 358, loss=0.01193985715508461
epoch 0, step 340 / 358, loss=0.01688408851623535
epoch 1, step 0 / 358, loss=0.030533963814377785
epoch 1, step 20 / 358, loss=0.017918717116117477
epoch 1, step 40 / 358, loss=0.03703705221414566
epoch 1, step 60 / 358, loss=0.013068962842226028
epoch 1, step 80 / 358, loss=0.013278307393193245
epoch 1, step 100 / 358, loss=0.008637825958430767
epoch 1, step 120 / 358, loss=0.010797485709190369
epoch 1, step 140 / 358, loss=0.007710881065577269
epoch 1, step 160 / 358, loss=0.007644436322152615
epoch 1, step 180 / 358, loss=0.015620463527739048
epoch 1, step 200 / 358, loss=0.013507158495485783
epoch 1, step 220 / 358, loss=0.0266572292894125
epoch 1, step 240 / 358, loss=0.007559360470622778
epoch 1, step 260 / 358, loss=0.008456951938569546
epoch 1, step 280 / 358, loss=0.006380652543157339
epoch 1, step 300 / 358, loss=0.0073578208684921265
epoch 1, step 320 / 358, loss=0.009011362679302692
epoch 1, step 340 / 358, loss=0.03037981316447258
```



## 7th Week
Document what have been done. Start to write a journal about how your learn the Neural Network from the very beginning, which could help anyone interested in machine learning and artificial intelligence. You could write your journal using markdown format.

You could also watch a few videos, which will help you to understand more about application of Neural Network.

- "Machine Learning Zero to Hero (Google I/O'19)", https://www.youtube.com/watch?v=VwVg9jCtqaU, you only need to watch the part presented by "Laurence", who is a famous engineer.

- "How we teach computers to understand pictures | Fei Fei Li", https://www.youtube.com/watch?v=40riCqvRoMs. It's presented by a very famous professor at Stanford.

## 8th Week
* Document what have been done
* 

## 9th Week
In this week, we will improve the accuracy using dice loss.

When we use BCEWithLogitsLoss, even if we train with more than 40 epoches, we only get about 35% accuracy when we test the accuracy using IoU (Intersection over Union).

The following resources explains the Dice Coefficient and we will use Dice Loss instead of BCEWithLogitsLoss for the loss function.

- "Image Segmentation Loss: IoU vs Dice Coefficient", https://www.youtube.com/watch?v=AZr64OxshLo, has very good explaination for the dice coefficient.
- "Loss Function Reference for Keras & PyTorch", https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch, has sample implementation for Dice Loss.
- https://www.kaggle.com/soulmachine/siim-deeplabv3

We will compare
- Binary Cross Entropy Loss, which is what we have used so far
- Dice Loss
- Dice Binary Cross Entropy Loss, which is the combination of dice loss and binary cross entropy loss.

The DiceLoss could be defined as
```python
from enum import Enum

class LossMethod(Enum):
    Bce = 0
    Dice = 1
    DiceBce = 2

# define a global variable to indicate which loss method to use
loss_method = LossMethod.DiceBce

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets):
        # The first dimension is the batch_size
        N = targets.size(0)
        inputs = torch.sigmoid(inputs)
        # flattern labels
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)

        EPSILON = 1e-7

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + EPSILON)/(union + EPSILON)
        return 1 - dice/N

dice_criterion = DiceLoss()
bce_criterion = nn.BCEWithLogitsLoss()

def calculate_loss(predicts, labels):
    # convert labels to float() as BCEWithLogitsLoss requires float numbers
    labels = labels.float()

    # predicts size is [batch_size, 2, W, H]
    # We only need to use the defect class to calculate the IoU
    # Get the predicted defect class
    defect_preds = predicts[:, 1, :, :]
    # Get the targeted defect class
    defect_targets = labels[:, 1, :, :]

    if loss_method == LossMethod.Bce:
        bce_loss = bce_criterion(predicts, labels)
        loss = bce_loss
        return loss
    elif loss_method == LossMethod.Dice:
        dice_loss = dice_criterion(defect_preds, defect_targets)
        loss = dice_loss 
        return loss
    else:
        # loss_method == LossMethod.DiceBce:
        dice_loss = dice_criterion(defect_preds, defect_targets)
        bce_loss = bce_criterion(predicts, labels)
        # After train about 30 epoches, the dice_loss is in the range of 0.5 and bce_loss is in the range of 0.01
        # we will divide the dice_loss by 20 times to make them equally weight
        loss = bce_loss + dice_loss / 20.0

        return loss

```

In the above code, the `inputs.view(N, -1)` is to flatten the matrix. For example, if the `inputs` is `[[1,2,3],[4,5,6]]`, the `inputs.view(-1)` means to change it to one dimensional and it becomes `[1,2,3,4,5,6]`.

As for why there is `EPSILON = 1e-7`, it's to avoid divide by zero when both inputs and targets are zero.

We will use
```python
    loss = calculate_loss(predicts, labels)
```
to calcualte the loss

The rest code will be same.

When use DiceLoss(), you will find that it may take a little more time to finish one epoch.

The following is the new code
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import os
import matplotlib.pyplot as plt
from enum import Enum

class LossMethod(Enum):
    Bce = 0
    Dice = 1
    DiceBce = 2


# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")

batch_size = 2
# make the size smaller so that we could run it quickly
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.0001
num_epochs = 20
# define a global variable to indicate which loss method to use
loss_method = LossMethod.Dice

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.src_img_names = os.listdir(src_image_folder)
        mod = len(self.src_img_names) % batch_size
        # remove the entries that not fall into the batch
        self.src_img_names = self.src_img_names[0: len(self.src_img_names) - mod]
        print(len(self.src_img_names))


    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc).convert("RGB")

        preprocess = transforms.Compose([
                                    transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc).convert("L")
        label_image_resized = trf_resize(label_image)
        # create numpy data from the image
        label_image_data = np.array(label_image_resized)
        # we will then create two classes, the first is the background and the second is the label.
        defect_class = np.zeros_like(label_image_data)
        defect_class[label_image_data > 0] = 1.0
        # background class
        background_class = np.zeros_like(label_image_data)
        background_class[label_image_data == 0] = 1.0
        label_image_classes = torch.tensor([background_class, defect_class])

        # return
        # src_image_data: [3, W, H]. 3 means 3 channels
        # label_image_classes: [2, W, H]. The first element is background_class, the second element is defect_class
        return src_image_data, label_image_classes


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    """
    x is [batch_size, 3, W, H]
    return is [batch_size, 2, W, H], where [batch_size, 0] is background_class, [batch_size, 1] is defect_class
    """
    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets):
        N = targets.size(0)
        inputs = torch.sigmoid(inputs)
        # flattern labels
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)

        EPSILON = 1e-7

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + EPSILON)/(union + EPSILON)
        return 1 - dice/N


dice_criterion = DiceLoss()
bce_criterion = nn.BCEWithLogitsLoss()

def calculate_loss(predicts, labels):
    # convert labels to float() as BCEWithLogitsLoss requires float numbers
    labels = labels.float()

    # predicts size is [batch_size, 2, W, H]
    # We only need to use the defect class to calculate the IoU
    # Get the predicted defect class
    defect_preds = predicts[:, 1, :, :]
    # Get the targeted defect class
    defect_targets = labels[:, 1, :, :]

    if loss_method == LossMethod.Bce:
        bce_loss = bce_criterion(predicts, labels)
        loss = bce_loss
        return loss
    elif loss_method == LossMethod.Dice:
        dice_loss = dice_criterion(defect_preds, defect_targets)
        loss = dice_loss 
        return loss
    else:
        # loss_method == LossMethod.DiceBce:
        dice_loss = dice_criterion(defect_preds, defect_targets)
        bce_loss = bce_criterion(predicts, labels)
        # After train about 30 epoches, the dice_loss is in the range of 0.5 and bce_loss is in the range of 0.01
        # we will divide the dice_loss by 20 times to make them equally weight
        loss = bce_loss + dice_loss / 20.0

        return loss


def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)

    fig, axarr= plt.subplots(2,3)

    # the image has three channels
    axarr[0,0].imshow(src_image[0], cmap = "gray")
    axarr[0,1].imshow(src_image[1], cmap = "gray")
    axarr[0,2].imshow(src_image[2], cmap = 'gray')

    # the labels has two, the first element is background, the second element is the label
    axarr[1,0].imshow(label_image[0], cmap="gray")
    axarr[1,1].imshow(label_image[1], cmap="gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SegmentationModel()

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    predicts = model.forward(inputs)
    print(predicts.shape)
    print(labels.shape)

    loss = calculate_loss(predicts, labels)
    print(loss)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loop(train_loader, model, optimizer)

def load_checkpoint_and_train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    checkpoint = torch.load(os.path.join(current_location, f"saved-deeplab-{loss_method.name}-checkpoint.pth"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loop(train_loader, model, optimizer)


def train_loop(train_loader, model, optimizer):
    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            predicts = model(images)
            loss = calculate_loss(predicts, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #    break

    torch.save(model, os.path.join(current_location, f"saved-deeplab-{loss_method.name}.pth"))
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, os.path.join(current_location, f"saved-deeplab-{loss_method.name}-checkpoint.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, f"saved-deeplab-{loss_method.name}.pth"))
    model.eval()
    with torch.no_grad():
        # only show the one that we are interesting
        # 20, 30 are OK, but
        # 40, 42 are not OK.
        # For testdata: 386, 396 are very bad
        target_index = 387
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)
                print(f"max={output.max()}")
                print(f"min={output.min()}")

                loss = calculate_loss(output, label)
                print(f"loss={loss}")

                output_raw = output.squeeze(0)
                print(f"max={output_raw.max()}")
                print(f"min={output_raw.min()}")

                output_background = output_raw[0]
                output_defect = output_raw[1]

                # the index who has the max value. As we only have two classes, it's either 0 or 1
                output_label = output_raw.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # draw the images
                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                target_label = label.squeeze(0)[1]

                intersection = output_label.logical_and(target_label)
                union = output_label.logical_or(target_label)
                accuracy = 1.0
                if union.sum().item() != 0:
                    accuracy = intersection.sum().item() / union.sum().item()
                print(f"index={index}: accuracy={accuracy}")


                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                axarr[0,1].imshow(torch.sigmoid(output_raw[0]), cmap="gray")

                # defect
                axarr[0,2].imshow(torch.sigmoid(output_raw[1]), cmap="gray")

                # expected label
                axarr[1,0].imshow(target_label, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                # show ytest, it should be same as output_label
                # axarr[1,2].imshow(ytest, cmap="gray")
                plt.show()


            index += 1

def test_model_accuracy():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, f"saved-deeplab-{loss_method.name}.pth"))
    model.eval()
    with torch.no_grad():
        index = 0
        accuracy_list = []

        for image, label in test_loader:
            output = model(image)

            loss = calculate_loss(output, label)
            print(f"loss={loss}")

            output_raw = output.squeeze(0)

            # the index who has the max value. As we only have two classes, it's either 0 or 1
            output_label = output_raw.argmax(dim=0)

            target_label = label.squeeze(0)[1]

            intersection = output_label.logical_and(target_label)
            union = output_label.logical_or(target_label)
            accuracy = 1.0
            if union.sum().item() != 0:
                accuracy = intersection.sum().item() / union.sum().item()
            accuracy_list.append(accuracy)
            print(f"index={index}: accuracy={accuracy}")

            index = index + 1

        accuracy_list_np = np.array(accuracy_list)
        print(f"Accuracy: min={accuracy_list_np.min()}, max={accuracy_list_np.max()}, average={np.average(accuracy_list_np)}")
        print(f"The least accuracy index={accuracy_list_np.argmin()}")



if __name__ == "__main__":
    print(f"loss_method={loss_method.name}")
    # test_imagedataset()
    # test_train_one()
    # train_and_save_model()
    load_checkpoint_and_train_and_save_model()
    # test_model()
    # test_model_accuracy()


```

When we use the new code, the following is the accuracy result:

|                       | IoU Accuracy using Testing Data   | IoU Accuracy using Training Data  |
| ---                   | ---                               | ---                               |
| BCE (42 epoches)      | 0.352                             | 0.638                             |
| Dice (40 epoches)     | 0.359                             | 0.627                             |
| Dice BCE (40 epoches) | 0.366                             | 0.697                             |

There are quite a few images in the testing data whose defects were not labeled correctly. It caused the acccuracy to be zero when we calculate the Intersection Over Union accuracy. For example, the testing data at index 386, 396.

# Summary
Software Engineer Intern

Use Python and PyTorch library to develop a solution to label product defects from image data using Convolutional Neural Network. I started by implementing the simplest nenural network, Perceptron, algorithm from scratch using pure Python. I then implemented a neural network with one hidden layer from scratch using pure Python and NumPy, and trained the model with MNIST dataset to recognize hand-written digits. Those implementation helped me to have deep understanding of Feed Forward and Backward Propagation algorithm. I then build a model using exsting PyTorch DeeplabV3 Residual Network, trained model with defect image data and used the model to lable the defect of product.  Applied different loss function and calculated the accuracy using Intersection Over Union.

## FAQ
### Why
The company already uses DeepLabV3 archtecture to build their solution to detect the defect. They have training data and they use their traning data to train the model, then use the model to detect the defect. They want to evaluate the other deep learning architecture. Also, as the PyTorch framework become more popular, the company want to try to build their solution on top of PyTorch framework. So this intern project was to build a model using DeepLabV3 model provided by PyTorch framework, train it with their existing dataset and use it to detect the defect.

### What is the basic principal of Neural Network
The basic principal of Neural Network consists of two process, Feed Forward and Backward Propagation.

The output is calculated using the the weights and bias along with the inputs. It's called Feed Forward. Then we use gradient decent and calcualte the derivative of the error/loss over the weights and bias, and use it adjust the weights and bias. This process is called Backward Propagation.

### What's the part you do not like?
The time to train a model is very long.
