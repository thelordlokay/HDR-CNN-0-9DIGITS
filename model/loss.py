import numpy as np

# Define cross-entropy loss function
def cross_entropy(inputs, labels):
    # Get the number of output classes
    out_num = labels.shape[0]
    # Compute the dot product between the predicted probabilities (inputs) and the ground truth labels
    p = np.sum(labels.reshape(1, out_num) * inputs)
    # Compute the negative log probability
    loss = -np.log(p)
    # Return the computed loss
    return loss
