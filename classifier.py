import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


labels_map = {
    "close_up" : 0,
    "medium" : 1,
    "wide" : 2
}
def feature_extract(img):
    root = os.getcwd()
    img_path = os.path.join(root, 'data', 'training_data', img)
    print(f"Extracting featuress from {img_path}")

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
    max_thres = 255
    min_thres = 255 / 3
    edges = cv.Canny(gray_frame, min_thres, max_thres)
    edge_pixels = np.sum(edges == 255)
    edge_density = edge_pixels / total_pixels
    print(edge_density)

    # average brightness
    gray_max = 255.0
    avg_bright = np.mean(gray_frame) / gray_max #scale 0.0 to 1.0
    print(avg_bright)

    # face detector
    # https://github.com/informramiz/Face-Detection-OpenCV
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

def main():
    with open("training_labels_clean.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    X = []
    y = []
    for item in training_data:
        img = item["file_name"]
        x_i = feature_extract(img)
        y_i = labels_map[item["shot_type"]] #y__binary = np.array([1 if label==0 else 0 for label in y])
        X.append(x_i)
        y.append(y_i)
    X = np.array(X)
    y = np.array(y)

if __name__ == "__main__":
    main()


