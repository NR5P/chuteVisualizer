import time, cv2
import numpy as np

class MotionDetector():
    def __init__(self):
        self.lastMotionDetected = None
        self.lastImg = None

    def detectMotion(self, currImg, triggerAreaBox, rectangleStarted):
        if triggerAreaBox[0] != 0 and triggerAreaBox[1] != 0 and rectangleStarted == False:
            xStart = triggerAreaBox[0][0]
            yStart = triggerAreaBox[0][1]
            xEnd = triggerAreaBox[1][0]
            yEnd = triggerAreaBox[1][1]
            width = xEnd - xStart
            height = yEnd - yStart
            triggerAreaImg = currImg[yStart:yStart + width,xStart:xStart + width]
            self.lastImg = self.lastImg[yStart:yStart + width,xStart:xStart + width]
            frameDelta = cv2.absdiff(triggerAreaImg, self.lastImg)
            self.lastImg = currImg
            if np.average(frameDelta) > 5:
                return True
        else:
            self.lastImg = currImg
