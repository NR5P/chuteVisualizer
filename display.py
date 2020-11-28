import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2

class Display():
    def __init__(self):
        self.VID_HEIGHT = 480
        self.VID_WIDTH = 640
        self.triggerBtnPressed = False
        self.captureBtnPressed = False

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
        print("Depth Scale is: " , depth_scale)

        # remove objects over certain distance
        clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object, align images
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.createButtons()

        cv2.namedWindow("chute", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("chute",self.process_click)


    def createButtons(self):
        # create buttons on right of image
        self.buttonImg = np.zeros((self.VID_HEIGHT, 200, 3), np.uint8)
        self.buttonImg[...] = 255
        self.setBtnColor()

    def process_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > self.VID_WIDTH and y < self.VID_HEIGHT / 2:
                self.handleCaptureAreaButton()
            elif x > self.VID_WIDTH and y > self.VID_HEIGHT / 2:
                self.handleTriggerAreaButton()
            elif x < self.VID_WIDTH and self.triggerBtnPressed:
                self.handleTriggerAreaBox()
            elif x < self.VID_WIDTH and self.captureBtnPressed:
                self.handleCaptureAreaBox()

    def handleTriggerAreaBox(self):
        pass

    def handleCaptureAreaBox(self):
        pass

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
        self.setBtnColor()

    def handleTriggerAreaButton(self):
        self.captureBtnPressed = False
        self.triggerBtnPressed = not self.triggerBtnPressed
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

    def display(self, image):
        imgWithButtons = np.hstack((image,self.buttonImg))
        cv2.imshow("chute", imgWithButtons)

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

                # Remove background - Set pixels further than clipping_distance to grey
                white_color = 255
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), white_color, color_image)

                threshold_img = self.preProcessing(bg_removed)
                contours, hierarchy = cv2.findContours(threshold_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                #cv2.imshow('threshold', threshold_img)

                self.display(color_image)

                # Render images
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                #cv2.drawContours(color_image, contours, -1, (0,255,0), 3)
                #images = np.hstack((color_image, depth_colormap))
                #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                #cv2.imshow('Align Example', images)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.pipeline.stop()