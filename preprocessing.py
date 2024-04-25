import numpy as np  # Importing numpy library for numerical operations
import cv2  # Importing OpenCV library for image processing
from scipy import ndimage  # Importing ndimage module from scipy for image operations
import math  # Importing math module for mathematical operations

# Function to shift the image by specified values along x and y axes
def shift(img,sx,sy):
    # Extracting the number of rows and columns from the image
    rows,cols = img.shape
    # Creating a 2x3 transformation matrix for shifting
    M = np.float32([[1,0,sx],[0,1,sy]])
    # Applying the affine transformation to shift the image
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted  # Returning the shifted image

# Function to calculate the best shift for centering the image
def getBestShift(img):
    # Calculating the center of mass of the image
    cy,cx = ndimage.measurements.center_of_mass(img)

    # Extracting the number of rows and columns from the image
    rows,cols = img.shape
    # Calculating the shift required to center the image
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty  # Returning the shift values

# Function to preprocess the input image
def preprocess(img):
    # Converting the input image into a numpy array and inverting its colors
    img=255-np.array(img).reshape(28,28).astype(np.uint8)
    # Applying Otsu's thresholding to binarize the image
    (thresh, gray) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Removing empty rows and columns from the top, left, bottom, and right sides of the image
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    # Resizing the image to fit into a 20x20 square while preserving aspect ratio
    rows,cols = gray.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    # Padding the resized image to fit into a 28x28 square
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    # Calculating the best shift to center the image
    shiftx,shifty = getBestShift(gray)
    # Shifting the image to the calculated position
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    # Reshaping the preprocessed image to match the input shape required by the model
    img = gray.reshape(1,28,28).astype(np.float32)

    # Normalizing the pixel values of the image
    img-= int(33.3952)
    img/= int(78.6662)
    return img  # Returning the preprocessed image
