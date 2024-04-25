import numpy as np

class Convolution2D:
    # Initialize the Convolution2D layer with specified parameters.
    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # Set the number of filters, kernel size, input channels, padding, stride, learning rate, and name.
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        # Initialize weights and bias with random values.
        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0, self.F):
            self.weights[i, :, :, :] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)),
                                                        size=(self.C, self.K, self.K))

        # Set padding, stride, learning rate, and name.
        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name

    # Apply zero padding to the input.
    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out

    # Perform forward pass through the Convolution2D layer.
    def forward(self, inputs):
        # Get dimensions of the input.
        C = inputs.shape[0]
        W = inputs.shape[1] + 2 * self.p
        H = inputs.shape[2] + 2 * self.p
        
        # Apply zero padding to the input.
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c, :, :] = self.zero_padding(inputs[c, :, :], self.p)
        
        # Calculate output dimensions.
        WW = int((W - self.K) / self.s + 1)
        HH = int((H - self.K) / self.s + 1)
        
        # Initialize feature maps.
        feature_maps = np.zeros((self.F, WW, HH))
        
        # Convolve input with filters.
        for f in range(self.F):
            for w in range(0, WW, self.s):
                for h in range(0, HH, self.s):
                    feature_maps[f, w, h] = np.sum(self.inputs[:, w:w+self.K, h:h+self.K]*self.weights[f, :, :, :]) + self.bias[f]
        return feature_maps

        # Perform backward pass through the Convolution2D layer.
    def backward(self, dy):
        # Get dimensions of the input.
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        # Get dimensions of the output.
        F, W, H = dy.shape
        
        # Compute gradients.
        for f in range(F):
            for w in range(0, W, self.s):
                for h in range(0, H, self.s):
                    dw[f, :, :, :] += dy[f, w, h] * self.inputs[:, w:w+self.K, h:h+self.K]
                    dx[:, w:w+self.K, h:h+self.K] += dy[f, w, h] * self.weights[f, :, :, :]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        # Update weights and bias using gradients and learning rate.
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    # Extract weights and bias for saving the model.
    def extract(self):
        return {self.name+'.weights': self.weights, self.name+'.bias': self.bias}

    # Set weights and bias for loading the model.
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Maxpooling2D:
    # Initialize the Maxpooling2D layer with pool size, stride, and name.
    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    # Perform forward pass through the Maxpooling2D layer.
    def forward(self, inputs):
        # Store the input for backward pass.
        self.inputs = inputs
        # Get dimensions of the input.
        C, W, H = inputs.shape
        # Calculate dimensions of the output.
        new_width = int((W - self.pool) / self.s + 1)
        new_height = int((H - self.pool) / self.s + 1)
        # Initialize output.
        out = np.zeros((C, new_width, new_height))
        # Apply max pooling.
        for c in range(C):
            for w in range(W // self.s):
                for h in range(H // self.s):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out

    # Perform backward pass through the Maxpooling2D layer.
    def backward(self, dy):
        # Get dimensions of the input.
        C, W, H = self.inputs.shape
        # Initialize gradient of input.
        dx = np.zeros(self.inputs.shape)
        # Compute gradients using max pooling indices.
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c, w:w+self.pool, h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w//self.pool, h//self.pool]
        return dx

    # Extract method for consistency with other layers (currently empty).
    def extract(self):
        return


class FullyConnected:
    # Initialize the FullyConnected layer with number of inputs, number of outputs, learning rate, and name.
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        # Initialize weights and bias with random values.
        self.weights = 0.01 * np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate  # Learning rate
        self.name = name  # Name of the layer

    # Perform forward pass through the FullyConnected layer.
    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for backward pass
        return np.dot(self.inputs, self.weights) + self.bias.T  # Compute output

    # Perform backward pass through the FullyConnected layer.
    def backward(self, dy):
        if dy.shape[0] == self.inputs.shape[0]:  # Check if dy needs to be transposed
            dy = dy.T
        # Compute gradients
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)
        # Update weights and bias
        self.weights -= self.lr * dw.T
        self.bias -= self.lr * db
        return dx  # Return gradient of inputs

    # Extract weights and bias for serialization.
    def extract(self):
        return {self.name+'.weights': self.weights, self.name+'.bias': self.bias}

    # Set weights and bias from pretrained values.
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Flatten:
    # Initialize Flatten layer (no parameters to initialize).
    def __init__(self):
        pass

    # Perform forward pass through the Flatten layer.
    def forward(self, inputs):
        # Save input shape for backward pass
        self.C, self.W, self.H = inputs.shape
        # Flatten the input tensor into a 1D vector
        return inputs.reshape(1, self.C*self.W*self.H)

    # Perform backward pass through the Flatten layer.
    def backward(self, dy):
        # Reshape the gradient to match the original input shape
        return dy.reshape(self.C, self.W, self.H)

    # Extract information (if any) from the Flatten layer (not implemented in this case).
    def extract(self):
        return  # No information to extract


class ReLu:
    # Initialize ReLU activation layer (no parameters to initialize).
    def __init__(self):
        pass

    # Perform forward pass through the ReLU activation layer.
    def forward(self, inputs):
        # Save the inputs for later use in backward pass
        self.inputs = inputs
        # Apply ReLU activation element-wise
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret

    # Perform backward pass through the ReLU activation layer.
    def backward(self, dy):
        # Copy the gradient to avoid modifying the original data
        dx = dy.copy()
        # Zero-out gradients where inputs were negative during forward pass
        dx[self.inputs < 0] = 0
        return dx

    # Extract information (if any) from the ReLU activation layer (not implemented in this case).
    def extract(self):
        return  # No information to extract

class Softmax:
    # Initialize Softmax activation layer (no parameters to initialize).
    def __init__(self):
        pass

    # Perform forward pass through the Softmax activation layer.
    def forward(self, inputs):
        # Compute exponential of inputs element-wise
        exp = np.exp(inputs)
        # Compute Softmax probabilities
        self.out = exp / np.sum(exp)
        return self.out

    # Perform backward pass through the Softmax activation layer.
    def backward(self, dy):
        # Compute the gradient of the Softmax layer with respect to the inputs
        # The derivative of the Softmax function is: Softmax(x) * (1 - Softmax(x))
        # However, since the derivative of cross-entropy loss with respect to Softmax input is: Softmax - ground_truth
        # We can directly subtract dy (ground truth) from the output of Softmax
        return self.out.T - dy.reshape(dy.shape[0], 1)

    # Extract information (if any) from the Softmax activation layer (not implemented in this case).
    def extract(self):
        return  # No information to extract










