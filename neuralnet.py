import numpy as np
import util
import matplotlib.pyplot as plt


class ThreeLayerNet:
    def __init__(self):
        self.W1 = None # first layer of weights
        self.b1 = None # first layer bias
        self.W2 = None # second layer of weights
        self.b2 = None # second layer bias
    
    def init_weights_task1(self):
        input_dim = 2
        hdim = 3
        output_dim = 2
        
        np.random.seed(0)
        self.W1 = np.random.randn(input_dim, hdim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hdim))
        self.W2 = np.random.randn(hdim, output_dim) / np.sqrt(hdim)
        self.b2 = np.zeros((1, output_dim))

    def softmax(z):
        return (np.exp(z)/np.sum(np.exp(z)))

    def forward_prop(self, X):
        """
        Forward pass given input
        :param x: input
        :return: returns probabilities of each class
        """
        softmax = lambda z:np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]
        Zh = np.dot(X, self.W1) + self.b1
        H1 = np.tanh(Zh)
        H2 = np.dot(H1, self.W2) + self.b2
        probs = softmax(H2)
        
        return probs

    def predict(self, x):
        """
        Predict an output (0 or 1)
        
        :param x: input
        :return: returns class with highest probability
        """
        # Forward propagation
        probs = self.forward_prop(x)
        
        return np.argmax(probs, axis=1)

    def calculate_loss(self, X,Y, reg_lambda=0.01):
        """
        Evaluate the total loss on the dataset
        
        :param X: input features
        :param Y: outputs
        :param reg_lambda: regularization strength
        """
        W1, b1, W2, b2 = self.W1,self.b1,self.W2,self.b2
        # Compute output probabilities
        probs = self.forward_prop(X)
        
        num_examples = len(X)
        # Compute the loss
        corect_logprobs = -np.log(probs[range(num_examples), Y])
        data_loss = np.sum(corect_logprobs)
        # Optional regularization
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss



    def train(self, X,Y, input_dim=2, output_dim=2, hdim=3, eta=0.01, num_passes=20000, reg_lambda=0.01, print_loss=False):
        """
        Learn parameters for the neural network and return the model.
        
        :param hdim: Number of nodes in the hidden layer
        :param output_dim: output layer dimensionality
        :param input_dim: input layer dimensionality
        :param eta: learning rate for gradient descent
        :param num_passes: Number of passes through training data for gradient descent
        :param reg_lambda: regularization strength
        :param print_loss: If True, print the loss every 1000 iterations
        """

        # random initialization of the network parameters
        np.random.seed(0)
        W1 = np.random.randn(input_dim, hdim) / np.sqrt(input_dim)
        b1 = np.zeros((1, hdim))
        W2 = np.random.randn(hdim, output_dim) / np.sqrt(hdim)
        b2 = np.zeros((1, output_dim))
        
        num_examples = len(X)
        
        iterations = []
        losses = []
        
        # gradient descent
        for i in range(0, num_passes):

            # Forward propagation
            softmax = lambda z:np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]
            Zh = np.dot(X, W1) + b1
            H1 = np.tanh(Zh)
            H2 = np.dot(H1, W2) + b2
            probs = softmax(H2)

            # Backpropagation
            e = np.column_stack((1-Y,Y))
            dZ2= probs - e

            dW2 = np.dot(np.transpose(H1), dZ2)
            db2 = np.sum(dZ2, axis=0)
            dZ1 = np.multiply(1 - np.power(H1, 2),  np.dot(dZ2,np.transpose(W2)))
            dW1 = np.dot(np.transpose(X), dZ1)
            db1 = np.sum(dZ1, axis=0)

            # Regularization
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # update parameters
            W1 += -eta * dW1
            b1 += -eta * db1
            W2 += -eta * dW2
            b2 += -eta * db2
            
            # store the parameters
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            
            # compute and print the loss
            loss = self.calculate_loss(X,Y)
            iterations.append(i)
            losses.append(float(loss))
            if print_loss and i % 1000 == 0:
              print("Iteration: %i, Loss: %f" %(i, loss))
        return iterations,losses


    def plot_decision_boundary(self,X,Y):
        """
        plot our decision boundary as a contour plot
        
        You don't need to undertand this method
        """
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)

