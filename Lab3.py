import numpy as np
import os
from PIL import Image

# Activation Functions
def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_derivative(dA, Z):
    dZ = np.where(Z <= 0, 0, dA)
    return dZ

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid_derivative(dA, Z):
    s = np.clip(sigmoid(Z)[0], 1e-8, 1-1e-8)  # Ensure `s` is not exactly 0 or 1
    dZ = dA * s * (1 - s)
    return dZ

# ActivationFunction Class
class ActivationFunction:
    def __init__(self, name='relu'):
        self.name = name
        if name == 'relu':
            self.forward = relu
            self.backward = relu_derivative
        elif name == 'sigmoid':
            self.forward = sigmoid
            self.backward = sigmoid_derivative
        else:
            raise ValueError('Unsupported activation function')

# Regularization Class
class Regularization:
    @staticmethod
    def l2_regularization_cost(layers, lambd):
        m = layers[0].A_prev.shape[1]
        L2_cost = sum(np.sum(np.square(layer.W)) for layer in layers)
        L2_cost *= (lambd / (2 * m))
        return L2_cost

    @staticmethod
    def l2_regularization_gradient(dW, W, lambd, m):
        return dW + (lambd / m) * W

# Dropout Class
class Dropout:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, A):
        self.mask = np.random.rand(*A.shape) < self.keep_prob
        A_drop = np.multiply(A, self.mask)
        A_drop /= self.keep_prob
        return A_drop

    def backward(self, dA):
        dA_drop = np.multiply(dA, self.mask)
        dA_drop /= self.keep_prob
        return dA_drop

# Layer Class (Updated with Dropout)
class Layer:
    def __init__(self, input_dim, output_dim, activation='relu', keep_prob=1.0):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((output_dim, 1))
        self.activation = ActivationFunction(activation)
        self.Z = None
        self.A_prev = None
        self.dW = None
        self.db = None
        self.dropout = Dropout(keep_prob) if keep_prob < 1.0 else None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, A_prev) + self.b
        A = self.activation.forward(self.Z)[0]
        if self.dropout:
            A = self.dropout.forward(A)
        return A

    def backward(self, dA, lambd):
        if self.dropout:
            dA = self.dropout.backward(dA)
        dZ = self.activation.backward(dA, self.Z)
        m = dA.shape[1]
        self.dW = np.dot(dZ, self.A_prev.T) / m
        self.dW += Regularization.l2_regularization_gradient(self.dW, self.W, lambd, m)  # L2 regularization
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev
    
class ForwardPropagation:
    @staticmethod
    def forward(layers, X):
        A = X
        caches = []
        for layer in layers:
            A_prev = A
            A = layer.forward(A_prev)
            caches.append((A_prev, layer.Z))
        return A, caches

class BackwardPropagation:
    @staticmethod
    def backward(layers, caches, Y, AL, lambd):
        gradients = {}
        dAL = AL - Y
        dA = dAL
        num_layers = len(layers)
        for l in reversed(range(num_layers)):
            current_cache = caches[l]
            layer = layers[l]
            dA_prev = layer.backward(dA, lambd)
            dA = dA_prev
            gradients["dW" + str(l+1)] = layer.dW
            gradients["db" + str(l+1)] = layer.db
        return gradients

class GradientDescentOptimizer:
    @staticmethod
    def update_parameters(layers, gradients, learning_rate):
        L = len(layers)
        for l in range(L):
            layers[l].W -= learning_rate * gradients["dW" + str(l+1)]
            layers[l].b -= learning_rate * gradients["db" + str(l+1)]

class MiniBatchGenerator:
    @staticmethod
    def create_mini_batches(X, Y, mini_batch_size=64, seed=0):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        
        # Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))
        
        # Partition (shuffled_X, shuffled_Y)
        num_complete_minibatches = m // mini_batch_size
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        
        return mini_batches

class NeuralNetwork:
# Modified to include learning rate scheduler and Adam optimizer
    def __init__(self, layer_dims, activation_funcs, keep_probs=None, lambd=0):
        self.layers = []
        self.lambd = lambd
        self.learning_rate = 0.0075
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Initializing time step for Adam optimizer
        if keep_probs is None:
            keep_probs = [1.0] * len(layer_dims)
        for i in range(1, len(layer_dims)):
            keep_prob = keep_probs[i-1]
            self.layers.append(Layer(layer_dims[i-1], layer_dims[i], activation_funcs[i-1], keep_prob))

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        AL = np.clip(AL, 1e-8, 1 - 1e-8)  # Ensure AL is never 0 or 1 when passed to log
        cross_entropy_cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        cross_entropy_cost = np.squeeze(cross_entropy_cost)
        
        L2_regularization_cost = Regularization.l2_regularization_cost(self.layers, self.lambd)
        total_cost = cross_entropy_cost + L2_regularization_cost
        return total_cost

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward_propagation(self, Y, AL):
        m = Y.shape[1]
        dAL = AL - Y  # Simplified gradient of cost with respect to AL
        dA = dAL
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            dA_prev = layer.backward(dA, self.lambd)
            dA = dA_prev

    def update_parameters_with_adam(self, gradients, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        L = len(self.layers)
        v_corrected = {}
        s_corrected = {}
        
        # Initialization for the first time step
        if t == 1:  # This ensures that the initialization happens once
            self.v = {key: np.zeros_like(value) for key, value in gradients.items()}
            self.s = {key: np.zeros_like(value) for key, value in gradients.items()}
        
        for l in range(L):
            # Moving average of the gradients
            self.v["dW" + str(l+1)] = (beta1 * self.v["dW" + str(l+1)]) + ((1-beta1) * gradients["dW" + str(l+1)])
            self.v["db" + str(l+1)] = (beta1 * self.v["db" + str(l+1)]) + ((1-beta1) * gradients["db" + str(l+1)])
            
            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l+1)] = self.v["dW" + str(l+1)] / (1 - beta1**t)
            v_corrected["db" + str(l+1)] = self.v["db" + str(l+1)] / (1 - beta1**t)
            
            # Moving average of the squared gradients
            self.s["dW" + str(l+1)] = (beta2 * self.s["dW" + str(l+1)]) + ((1-beta2) * np.square(gradients["dW" + str(l+1)]))
            self.s["db" + str(l+1)] = (beta2 * self.s["db" + str(l+1)]) + ((1-beta2) * np.square(gradients["db" + str(l+1)]))
            
            # Compute bias-corrected second raw moment estimate
            s_corrected["dW" + str(l+1)] = self.s["dW" + str(l+1)] / (1 - beta2**t)
            s_corrected["db" + str(l+1)] = self.s["db" + str(l+1)] / (1 - beta2**t)
            
            # Update parameters
            self.layers[l].W -= learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            self.layers[l].b -= learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    def train(self, X, Y, learning_rate=0.0075, num_iterations=2500):
        decay_rate = 0.95
        for i in range(num_iterations):
            self.t += 1
            AL, caches = ForwardPropagation.forward(self.layers, X)
            cost = self.compute_cost(AL, Y)
            gradients = BackwardPropagation.backward(self.layers, caches, Y, AL, self.lambd)
            self.update_parameters_with_adam(gradients, self.t, learning_rate, self.beta1, self.beta2, self.epsilon)
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
            
            learning_rate *= decay_rate  # Decay the learning rate

# Helper Function to Load Images
def load_images_from_directory(directory, image_size=(64, 64)):
    images, labels = [], []
    for label, subdir in enumerate(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in os.listdir(subdir_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(subdir_path, filename)
            image = Image.open(image_path).convert('L').resize(image_size)
            images.append(np.asarray(image).flatten() / 255.0)
            labels.append(label)
    return np.array(images).T, np.array(labels).reshape(1, -1)

# Example neural network setup and training
def example_neural_network():
    base_folder = "/Users/naiefabdullahshakil/Desktop/Face Mask Dataset/Train"
    X, Y = load_images_from_directory(base_folder)
    layer_dims = [4096, 10, 8, 8, 4, 1]
    activation_funcs = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    nn = NeuralNetwork(layer_dims, activation_funcs, lambd=0.1, keep_probs=[1.0, 0.8, 0.8, 1.0, 1.0])
    nn.train(X, Y, learning_rate=0.0075, num_iterations=2500)

if __name__ == "__main__":
    example_neural_network()