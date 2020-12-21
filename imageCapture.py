import time, cv2
import numpy as np

class ImageCapture():
    def __init__(self):
        self.lastImageCapture = None

    def saveImage(self, image: np.ndarray, bgRemoved: np.ndarray, depthImage: np.ndarray, pointCloudImg: np.ndarray):
        topImages = np.hstack((image,bgRemoved))
        bottomImages = np.hstack((depthImage,pointCloudImg))
        allImages = np.vstack((topImages, bottomImages))
        self.lastImageCapture = time.time()
        cv2.imwrite("images/"+self.lastImageCapture,allImages)

    def getLastImageCapture(self):
        return self.lastImageCapture
        self.lastImageCapture = None