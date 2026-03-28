import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax

labels_map = {
    "close_up" : 0,
    "medium" : 1,
    "wide" : 2
}

def feature_extract(img, testing=False):
    # extracting edge density, average brightness, and face fraction from datasets
    root = os.getcwd()
    if testing:
        folder = "testing_data"
    else:
        folder = "training_data"
    img_path = os.path.join(root, 'data', folder, img)
    print(f"Extracting features from {img_path}")

    #make frame gray
    frame = cv.imread(img_path)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    total_pixels = gray_frame.size

    # display frame
    '''
    cv.imshow("Image", gray_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''

    # edge density
    # Canny edge 3:1 ratio, then calculates density of white lines on black
    #
    max_thres = 255
    min_thres = 255 / 3
    edges = cv.Canny(gray_frame, min_thres, max_thres)
    edge_pixels = np.sum(edges == 255)
    edge_density = edge_pixels / total_pixels
    print(edge_density)

    # average brightness
    # average values of grayscale 0 to 255, normalized
    #
    gray_max = 255.0
    avg_bright = np.mean(gray_frame) / gray_max #scale 0.0 to 1.0
    print(avg_bright)

    # face detector
    # https://github.com/informramiz/Face-Detection-OpenCV
    # After detecting faces, calculate area of drawn box fraction of entire frame.
    # If no face found, fraction is 0.
    haar_face_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    faces = haar_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    print("Faces found: ", len(faces))
    if len(faces) > 0: 
        w = faces[0][2]
        h = faces[0][3]
        face_frac = w * h / total_pixels
        print(face_frac)
    else: 
        face_frac = 0.0
    return [edge_density, avg_bright, face_frac]

def one_hot_encode(y, num_classes):
    # one hot encoding for clean loss calculation
    return np.eye(num_classes)[y]

def compute_loss(X, Y, W, b, mu):
    # Equation take from 
    # https://medium.com/data-science/multiclass-logistic-regression-from-scratch-9cc0007da372
    # Added l2 regularization and bias term
    Z = X @ W + b
    N = X.shape[0]
    #shape NxC
    P = softmax(Z, axis=1)
    #loss = 1/N * (-np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    loss = -np.sum(Y * np.log(P)) / N
    loss += mu * np.sum(W**2)
    return loss

def compute_gradient(X, Y, W, b, mu):
    # Equation take from 
    # https://medium.com/data-science/multiclass-logistic-regression-from-scratch-9cc0007da372
    # Added l2 regularization and bias term
    N = X.shape[0]
    Z = X @ W + b
    P = softmax(Z, axis=1)
    dW = 1/N * (X.T @ (P - Y)) + 2 * mu * W
    db = 1/N * np.sum(P - Y, axis=0) 

    return dW, db

def predict(X, W, b):
    # Argmax to find largest probability per row
    #
    Z = X @ W + b
    P = softmax(Z, axis=1)
    return np.argmax(P, axis=1)

def prep_data(json_name, testing = False):
    # Preparing feature matrix and labels vector
    # X is R^NxM, N images, M features
    # Y = R^N, N images per class c
    root = os.getcwd()
    json_path = os.path.join(root, 'labels', json_name)
    with open(json_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    X = []
    y = []
    for item in training_data:
        img = item["file_name"]
        x_i = feature_extract(img, testing)
        y_i = labels_map[item["shot_type"]]
        X.append(x_i)
        y.append(y_i)
    X = np.array(X)
    y = np.array(y)

    return X, y

def main():

    X_train, y_train = prep_data("training_labels_clean.json")
    X_test, y_test = prep_data("testing_labels_clean.json", testing=True)

    # N = 3
    num_features = X_train.shape[1]
    # C = 3
    num_classes = len(labels_map)

    Y_train = one_hot_encode(y_train, num_classes)
    Y_test = one_hot_encode(y_test, num_classes)

    # W is initialized NxC with random small floats following Gaussian distirbution
    b = np.zeros(num_classes)
    W = 0.01 * np.random.randn(num_features, num_classes)

    #arbitrary selections
    mu = 0.001
    eta = 0.01
    epochs = 100000

    for epoch in range(epochs):

        loss = compute_loss(X_train, Y_train, W, b, mu)
        dW, db = compute_gradient(X_train, Y_train, W, b, mu)

        #update values
        W -= eta * dW
        b -= eta * db

        print(f"Epoch {epoch}, Loss: {loss}")

    # Training and testing evaluation
    y_pred_train = predict(X_train, W, b)
    train_accuracy = np.mean(y_pred_train == y_train)

    y_pred_test = predict(X_test, W, b)
    test_accuracy = np.mean(y_pred_test == y_test)

    print("Training accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)

    test_loss = compute_loss(X_test, Y_test, W, b, mu)
    print("Test loss:", test_loss)      

if __name__ == "__main__":
    main()


