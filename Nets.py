import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Dense:
    def __init__(self, layer_sizes:[], dropout_rate = None):
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.n_layers = len(layer_sizes)
        self.weights = []
        self.biases = []

        for i in range(self.n_layers-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        if self.dropout_rate == None:
            self.dropout_rate = [0.0] * self.n_layers
        elif isinstance(self.dropout_rate, float):
            self.dropout_rate = [self.dropout_rate] * self.n_layers
        elif len(self.dropout_rate) == self.n_layers - 2:
            self.dropout_rate = dropout_rate + [0.0]
        else:
            raise ValueError('Dropout rate must be either float or list of floats and match the number of layers')

        self.Z = []
        self.A = []
        self.masks = []


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

    def feedforward(self, X, training = True):
        """self.Z1 = np.dot(X, self.weights_input_first) + self.bias_input_first
        self.A1 = self.leaky_relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights_first_second) + self.bias_first_second
        self.A2 = self.leaky_relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.weights_second_output) + self.bias_second_output
        self.A3 = self.softmax(self.Z3)"""
        self.Z = []
        self.A = [X]
        self.masks = []
        for i in range(self.n_layers - 1):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(z)
            if i == self.n_layers - 2:
                a = self.softmax(z)
            else:
                a = self.leaky_relu(z)

                if self.dropout_rate[i] > 0 and training:
                    mask = np.random.binomial(1, 1-self.dropout_rate[i], size = a.shape)
                    self.masks.append(mask)
                    a *= mask
                    a /= (1-self.dropout_rate[i])
                else:
                    self.masks.append(None)

            self.A.append(a)

        return self.A[-1]

    def backpropagation(self, X, y, learning_rate):
        n_samples = X.shape[0]

        y_one_hot = np.zeros((n_samples, self.layer_sizes[-1]))
        y_one_hot[np.arange(n_samples), y] = 1

        dZ = self.A[-1] - y_one_hot
        dW = np.dot(self.A[-2].T, dZ) / n_samples
        db = np.sum(dZ, axis=0, keepdims=True)

        """dA2 = np.dot(dZ3, self.weights_second_output.T)
        dZ2 = dA2 * self.leaky_relu_derivative(self.Z2)
        dW1 = np.dot(self.A1.T, dZ2) / n_samples
        db1 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.weights_first_second.T)
        dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
        dW0 = np.dot(X.T, dZ1) / n_samples
        db0 = np.sum(dZ1, axis=0, keepdims=True)"""

        self.weights[-1] -= learning_rate * dW
        self.biases[-1] -= learning_rate * db

        for i in range(self.n_layers - 2, 0, -1):
            dA = np.dot(dZ, self.weights[i].T)

            if self.dropout_rate[i - 1] > 0 and self.masks[i - 1] is not None:
                dA *= self.masks[i - 1]
                dA /= (1 - self.dropout_rate[i - 1])

            dZ = dA * self.leaky_relu_derivative(self.Z[i-1])
            dW = np.dot(self.A[i-1].T, dZ) / n_samples
            db = np.sum(dZ, axis=0, keepdims=True)
            self.weights[i - 1] -= learning_rate * dW
            self.biases[i - 1] -= learning_rate * db

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