import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open("training_labels_clean.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)

labels_map = {
    "close_up" : 0,
    "medium" : 1,
    "wide" : 2
}

#edge density

#average brightness
frame = cv2.imread('./data/training_data/11.jpg')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_max = 255.0
avg_bright = np.mean(gray_frame) / gray_max #scale 0.0 to 1.0
print(avg_bright)
#display gray
#cv2.imshow("Image", gray_frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#face detector
'''face_cascade = cv2.CascadeClassifier("./data/haarcascade.xml")
#from geeksforgeeks.org
def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img
img = cv2.imread("./data/training_data/11.jpg")
result = adjusted_detect_face(img)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()'''