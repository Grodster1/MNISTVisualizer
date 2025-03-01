import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


class Dense:
    def __init__(self, n_input, n_output, first_layer_size, second_layer_size):
        self.n_input = n_input
        self.n_output = n_output
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size

        self.weights_input_first = np.random.randn(self.n_input, self.first_layer_size) * np.sqrt(2 / self.n_input)
        self.weights_first_second = np.random.randn(self.first_layer_size, self.second_layer_size) * np.sqrt(
            2 / self.first_layer_size)
        self.weights_second_output = np.random.randn(self.second_layer_size, self.n_output) * np.sqrt(
            2 / self.second_layer_size)

        self.bias_input_first = np.zeros((1, self.first_layer_size))
        self.bias_first_second = np.zeros((1, self.second_layer_size))
        self.bias_second_output = np.zeros((1, self.n_output))

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def softmax(self, x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y, y_hat):
        n_samples = y_hat.shape[0]
        y_one_hot = np.zeros_like(y_hat)
        y_one_hot[np.arange(n_samples), y] = 1
        log_likelihood = -np.sum(y_one_hot * np.log(y_hat + 1e-8))
        return log_likelihood / n_samples

    def feedforward(self, X):
        self.Z1 = np.dot(X, self.weights_input_first) + self.bias_input_first
        self.A1 = self.leaky_relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights_first_second) + self.bias_first_second
        self.A2 = self.leaky_relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.weights_second_output) + self.bias_second_output
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def backpropagation(self, X, y, learning_rate):
        n_samples = X.shape[0]

        y_one_hot = np.zeros((n_samples, self.n_output))
        y_one_hot[np.arange(n_samples), y] = 1

        dZ3 = self.A3 - y_one_hot
        dW2 = np.dot(self.A2.T, dZ3) / n_samples
        db2 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, self.weights_second_output.T)
        dZ2 = dA2 * self.leaky_relu_derivative(self.Z2)
        dW1 = np.dot(self.A1.T, dZ2) / n_samples
        db1 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.weights_first_second.T)
        dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
        dW0 = np.dot(X.T, dZ1) / n_samples
        db0 = np.sum(dZ1, axis=0, keepdims=True)

        self.weights_second_output -= learning_rate * dW2
        self.bias_second_output -= learning_rate * db2

        self.weights_first_second -= learning_rate * dW1
        self.bias_first_second -= learning_rate * db1

        self.weights_input_first -= learning_rate * dW0
        self.bias_input_first -= learning_rate * db0

    @staticmethod
    def generate_batches(X, y, batch_size):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]

    def train(self, X, y, epochs, learning_rate, batch_size=32):
        losses = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            for X_batch, y_batch in self.generate_batches(X, y, batch_size):
                output = self.feedforward(X_batch)
                loss = self.cross_entropy_loss(y_batch, output)
                self.backpropagation(X_batch, y_batch, learning_rate)
            losses.append(loss)
            #if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.6f}")
        return losses

    def predict_proba(self, x):
        return self.feedforward(x)

    def predict(self, x):
        proba = self.predict_proba(x)
        prediction = np.argmax(proba, axis=1)
        return prediction

    def evaluate(self, X_test, y_test):
        predictions_proba = self.feedforward(X_test)
        predicted_labels = self.predict(X_test)
        loss = self.cross_entropy_loss(y_test, predictions_proba)
        accuracy = accuracy_score(y_test, predicted_labels)
        print(f"Test loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return loss, accuracy

    def get_hidden_activations(self, X):
        self.feedforward(X)
        return self.A1, self.A2