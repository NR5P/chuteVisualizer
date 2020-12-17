import math
from Models.outerPoints import OuterPoints
from Models.dimensions import Dimensions

class Measurement:
    def __init__(self):
        self.widthFeet: float = None
        self.heightFeet: float = None
        self.cameraHeightPixels: int = 1080 
        self.cameraWidthPixels: int = 1920
        self.focalLengthMM: float = 1.88
        self.pixelSizeMM: float = .0014 
        self.distanceInches: int = None
        self.sensorHeightMM: float = self.pixelSizeMM * self.cameraHeightPixels
        self.sensorWidthMM: float = self.pixelSizeMM * self.cameraWidthPixels

    def setOuterPoints(self, outerPoints: OuterPoints, distance: int):
        self.outerPoints = outerPoints
        self.distanceInches = distance
        self.widthFeet = self.calculateWidth(outerPoints: OuterPoints)
        self.heightFeet = self.calculateHeight(outerPoints: OuterPoints)

    def calculateWidth(self, outerPoints: OuterPoints) -> float:
        objectWidthPixels = self.getDistanceBetweenPoints(outerPoints.left, outerPoints.right)
        objectWidthOnSensorMM = (self.sensorWidthMM * objectWidthPixels) / self.cameraWidthPixels
        realObjectWidthFeet = ((self.distanceInches / 12) * objectWidthOnSensorMM) / self.focalLengthMM
        return realObjectWidthFeet

    def calculateHeight(self, outerPoints: OuterPoints) -> float:
        objectHeightPixels = self.getDistanceBetweenPoints(outerPoints.top, outerPoints.bottom)
        objectHeightOnSensorMM = (self.sensorHeightMM * objectHeightPixels) / self.cameraHeightPixels
        realObjectHeightFeet = ((self.distanceInches / 12) * objectHeightOnSensorMM) / self.focalLengthMM
        return realObjectHeightFeet

    def getDistanceBetweenPoints(self, tupleA: Tuple[int, int], tupleB: tuple[int, int]) -> float:
        #TODO: the pixels are different because using lower res than camera
        difference = tuple(x-y for x,y in zip(tupleA,tupleB))
        D = difference[0] **2 + difference[1] **2
        return math.sqrt(D)

    def getHeightInches(self) -> int:
        return int(self.heightFeet * 12)

    def getWidthInches(self) -> int:
        return int(self.widthFeet * 12)

    def getDimensions(self) -> Dimensions:
        return Dimensions(self.getHeightInches(), self.getWidthInches())