import cv2
import numpy as np

# Function assumes that the image was passed to function correctly and is already in BGR format, not in RGB.
# Function return: image, boxes
def EdgeBoxes(image, display=False):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_img, 50, 250)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_copy = []
    if display:
        img_copy = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, w, h])
        if display:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if display:
        cv2.imshow('Edges', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, boxes

# EdgeBoxes(image=cv2.imread("venice.jpg"), display=True)