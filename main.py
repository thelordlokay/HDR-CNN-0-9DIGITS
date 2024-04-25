import struct
import numpy as np
from model.network import Net

DATASET_DIRECTORY = 'D:/Handwritten digit/dataset/'
LOG_FILE = 'training_log.txt'

def download_and_parse_mnist_file(filename):
    with open(filename, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        if magic == 2051:  # Images file
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, rows, cols)
        elif magic == 2049:  # Labels file
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError("Invalid magic number: %s" % magic)
    return data

def load_train_images():
    return download_and_parse_mnist_file(DATASET_DIRECTORY + 'train-images.idx3-ubyte')

def load_test_images():
    return download_and_parse_mnist_file(DATASET_DIRECTORY + 't10k-images.idx3-ubyte')

def load_train_labels():
    return download_and_parse_mnist_file(DATASET_DIRECTORY + 'train-labels.idx1-ubyte')

def load_test_labels():
    return download_and_parse_mnist_file(DATASET_DIRECTORY + 't10k-labels.idx1-ubyte')

def log(message):
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')

log('Loading data......')
num_classes = 10
train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

# Print dataset information
log('Dataset Information:')
log(f'Number of training samples: {len(train_images)}')
log(f'Number of testing samples: {len(test_images)}')
log(f'Shape of training images: {train_images.shape}')
log(f'Shape of testing images: {test_images.shape}')
log(f'Number of classes: {num_classes}')

# Preprocess the data (normalize)
train_images = (train_images - np.mean(train_images)) / np.std(train_images)
test_images = (test_images - np.mean(test_images)) / np.std(test_images)

# Reshape data for the network
training_data = train_images.reshape(-1, 1, 28, 28)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(-1, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]

# Initialize the neural network
net = Net()

# Train the network
log('Training Lenet......')
net.train(training_data, training_labels, batch_size=32, epoch=1, weights_file='pretrained_weights.pkl')

# Test the network
log('Testing Lenet......')
net.test(testing_data, testing_labels, test_size=100)

# Test the network with pretrained weights
log('Testing with pretrained weights......')
net.test_with_pretrained_weights(testing_data, testing_labels, test_size=100, weights_file='pretrained_weights.pkl')









