import time, cv2

class ImageCapture():
    def __init__(self):
        self.lastImageCapture = 0

    def saveImage(self, image, bgRemoved, depthImage, pointCloudImg):
        topImages = np.hstack((image,bgRemoved))
        bottomImages = np.hstack((depthImage,pointCloudImg))
        allImages = np.vstack((topImages, bottomImages))
        self.lastImageCapture = time.time()
        cv2.imwrite("images/"+self.lastImageCapture,allImages)

    def getLastImageCapture(self):
        return self.lastImageCapture
        self.lastImageCapture = 0