import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2, time
from motionDetector import MotionDetector
from imageCapture import ImageCapture

class Display():
    def __init__(self):
        self.VID_HEIGHT = 480
        self.VID_WIDTH = 640
        self.triggerBtnPressed = False
        self.captureBtnPressed = False
        self.captureAreaBox = [0] * 2
        self.triggerAreaBox = [0] * 2
        self.rectangleStarted = False
        self.clipping_distance = 1 # clipping distance in meters
        self.displayPointCloud = False
        self.pointCloudDecimate = 1
        self.pointCloudColor = True
        self.pointCloudImage = np.zeros((self.VID_HEIGHT,self.VID_WIDTH,3), np.uint8)
        self.blankImage = np.zeros((self.VID_HEIGHT,self.VID_WIDTH,3), np.uint8)
        self.motionDetector = MotionDetector()
        self.imageCapture = ImageCapture()

        # Create a pipeline
        self.pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.VID_WIDTH, self.VID_HEIGHT, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, self.VID_WIDTH, self.VID_HEIGHT, rs.format.bgr8, 15)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # remove objects over certain distance
        self.clipping_distance = self.clipping_distance / depth_scale

        # Create an align object, align images
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.createButtons()

        cv2.namedWindow("chute", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("chute",self.process_click)

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        # Processing blocks
        self.pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.pointCloudDecimate)
        self.colorizer = rs.colorizer()


    def createButtons(self):
        self.buttonImg = np.zeros((self.VID_HEIGHT, 200, 3), np.uint8)
        self.buttonImg[...] = 255
        self.setBtnColor()

    def process_click(self, event, x, y, flags, params):
        if x > self.VID_WIDTH and y < self.VID_HEIGHT / 2 and event == cv2.EVENT_LBUTTONDOWN:
            self.handleCaptureAreaButton()
        elif x > self.VID_WIDTH and y > self.VID_HEIGHT / 2 and event == cv2.EVENT_LBUTTONDOWN:
            self.handleTriggerAreaButton()
        elif x < self.VID_WIDTH and self.triggerBtnPressed:
            self.handleRectangleDraw(event, x, y, self.triggerAreaBox)
        elif x < self.VID_WIDTH and self.captureBtnPressed:
            self.handleRectangleDraw(event, x, y, self.captureAreaBox)

    def handleRectangleDraw(self, event, x, y, captureBox):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rectangleStarted = True
            captureBox[0] = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.rectangleStarted = False
            self.triggerBtnPressed = self.captureBtnPressed = False
            self.setBtnColor()
            captureBox[1] = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and self.rectangleStarted:
            captureBox[1] = (x,y)


    def setBtnColor(self):
        half = int(self.VID_HEIGHT/2)
        if self.captureBtnPressed:
            self.buttonImg[0:int(self.VID_HEIGHT/2),:,:] = 100
        else:
            self.buttonImg[0:int(self.VID_HEIGHT/2),:,:] = 255
        if self.triggerBtnPressed:
            self.buttonImg[int(self.VID_HEIGHT/2):self.VID_HEIGHT,:,:] = 100
        else:
            self.buttonImg[int(self.VID_HEIGHT/2):self.VID_HEIGHT,:,:] = 255
        self.buttonImg[half-5:half+5,:,:] = 0
        cv2.putText(self.buttonImg, 'Capture',(25,int(half/2)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
        cv2.putText(self.buttonImg, 'Area',(25,int(half/2+30)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
        cv2.putText(self.buttonImg, 'Trigger',(25,int(half*1.5)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
        cv2.putText(self.buttonImg, 'Area',(25,int(half*1.5+30)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)

    def handleCaptureAreaButton(self):
        self.triggerBtnPressed = False
        self.captureBtnPressed = not self.captureBtnPressed
        if self.captureBtnPressed == True:
            self.captureAreaBox = [0] * 2
        self.setBtnColor()

    def handleTriggerAreaButton(self):
        self.captureBtnPressed = False
        self.triggerBtnPressed = not self.triggerBtnPressed
        if self.triggerBtnPressed == True:
            self.triggerAreaBox = [0] * 2
        self.setBtnColor()

    def handleButtonPress(self, button):
        pass

    def preProcessing(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blured_img = cv2.GaussianBlur(gray_img,(5,5),1)
        canny_img = cv2.Canny(blured_img,200,200)
        kernel = np.ones((2,2))
        dilated = cv2.dilate(canny_img,kernel,iterations=2)
        preproc_img = cv2.erode(dilated,kernel,iterations=1)

        return preproc_img

    def display(self, image, bgRemoved, depthImage, pointCloudImg):
        blankImage = np.zeros((self.VID_HEIGHT,self.VID_WIDTH,3), np.uint8)
        blankImageButtons = np.zeros((self.VID_HEIGHT,200,3), np.uint8)
        imgWithButtons = np.hstack((image,bgRemoved,self.buttonImg))
        bottomImages = np.hstack((depthImage,pointCloudImg,blankImageButtons))
        entireScreen = np.vstack((imgWithButtons, bottomImages))
        if self.captureAreaBox[0] != 0 and self.captureAreaBox[1] != 0:
            cv2.rectangle(entireScreen, self.captureAreaBox[0], self.captureAreaBox[1], (0,255,0), thickness=1) 
        if self.triggerAreaBox[0] != 0 and self.triggerAreaBox[1] != 0:
            cv2.rectangle(entireScreen, self.triggerAreaBox[0], self.triggerAreaBox[1], (0,0,255), thickness=1) 
        cv2.imshow("chute", entireScreen)

    
    def project(self,v):
        """project 3d vector array to 2d"""
        h, w = self.pointCloudImage.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def pointcloud(self, out, verts, texcoords, color):
        proj = self.project(verts)

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]

        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]


    def getPointCloudImage(self,colorImage,colorFrame,depthFrame,pointCloudImage):
        depth_colormap = np.asanyarray(
            self.colorizer.colorize(depthFrame).get_data())
        if self.pointCloudColor:
            mapped_frame, color_source = colorFrame, colorImage
        else:
            mapped_frame, color_source = depthFrame, depth_colormap
        points = self.pc.calculate(depthFrame)
        self.pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        self.pointCloudImage.fill(0)

        self.pointcloud(pointCloudImage, verts, texcoords, color_source)

        return pointCloudImage
        
    def getRoiAndResize(self, image):            
        try:
            captureWidthStart = self.captureAreaBox[0][0]
            captureHeightStart = self.captureAreaBox[0][1]
            captureWidth = self.captureAreaBox[1][0] - captureWidthStart
            captureHeight = self.captureAreaBox[1][1] - captureHeightStart
            roiImage = image[captureHeightStart:captureHeightStart+captureHeight,captureWidthStart:captureWidthStart+captureWidth]

            r = self.VID_WIDTH / float(captureWidth)
            dim = (self.VID_WIDTH, int(captureHeight * r))
            roiImage = cv2.resize(roiImage,dim,interpolation=cv2.INTER_AREA)
            borderHeight = self.VID_HEIGHT - roiImage.shape[0]
            topBorder = int(borderHeight / 2)
            bottomBorder = borderHeight - topBorder
            roiImage = cv2.copyMakeBorder(roiImage, topBorder, bottomBorder, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
            return roiImage
        except Exception as e:
            print(e)
            return self.blankImage



    def displayLoop(self):
        try:
            while True:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                motionDetected = self.motionDetector.detectMotion(color_image, self.triggerAreaBox, self.rectangleStarted)
                if motionDetected == True:
                    # Remove background - Set pixels further than clipping_distance to grey
                    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                    white_color = 255
                    bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), white_color, color_image)

                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        
                    if self.displayPointCloud:
                        self.pointCloudImage = self.getPointCloudImage(bg_removed ,color_frame ,aligned_depth_frame, self.pointCloudImage)
                    else:
                        self.pointCloudImage = self.blankImage
                    self.display(color_image, self.getRoiAndResize(bg_removed), self.getRoiAndResize(depth_colormap), self.getRoiAndResize(self.pointCloudImage))
                else:
                    self.display(color_image, self.blankImage, self.blankImage, self.blankImage)

                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                # Press p to toggle point cloud
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord('p'):
                    self.displayPointCloud = not self.displayPointCloud
        finally:
            self.pipeline.stop()