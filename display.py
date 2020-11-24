import cv2
import numpy as np

class Display:
    def __init__(self, height):
        self.displayHeight = height

    def display(self, image):
        btnGui = self.createButtonWindow()
        imgWithButtons = np.hstack((image,btnGui))
        cv2.namedWindow("chute visializer", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("chute visializer", imgWithButtons)

    def createButtonWindow(self):
        img = np.zeros((self.displayHeight, 200, 3), np.uint8)
        half = int(self.displayHeight/2)
        img[...] = 255
        img[half-5:half+5,:,:] = 0
        cv2.putText(img, 'Button',(25,25),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
        return img
