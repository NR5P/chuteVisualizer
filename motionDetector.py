import time, cv2
import numpy as np
from typing import List, Tuple

class MotionDetector():
    def __init__(self):
        self.lastMotionDetected = None
        self.motionSensitivity = 10
        self.lastImg = None
        self.motionDelayTime = 1

    def detectMotion(self, currImg: np.ndarray, triggerAreaBox: List[Tuple[int, int]], rectangleStarted: bool) -> bool:
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
            if np.average(frameDelta) > self.motionSensitivity:
                if self.lastMotionDetected == None or time.time() - self.lastMotionDetected > self.motionDelayTime:
                    self.lastMotionDetected = time.time()
            if self.lastMotionDetected != None and time.time() - self.lastMotionDetected < 2:
                return True
        else:
            self.lastImg = currImg
        return False