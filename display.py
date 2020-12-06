import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2, time
from appState import AppState

class Display():
    def __init__(self):
        self.VID_HEIGHT = 480
        self.VID_WIDTH = 640
        self.triggerBtnPressed = False
        self.captureBtnPressed = False
        self.captureAreaBox = [0] * 2
        self.triggerAreaBox = [0] * 2
        self.lastImg = None
        self.rectangleStarted = False
        self.clipping_distance = 1 # clipping distance in meters
        self.state = AppState()
        self.pointCloudImage = np.zeros((self.VID_HEIGHT,self.VID_WIDTH,3), np.uint8)

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
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
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

    def detectMotion(self, currImg, lastImg):
        if self.triggerAreaBox[0] != 0 and self.triggerAreaBox[1] != 0 and self.rectangleStarted == False:
            xStart = self.triggerAreaBox[0][0]
            yStart = self.triggerAreaBox[0][1]
            xEnd = self.triggerAreaBox[1][0]
            yEnd = self.triggerAreaBox[1][1]
            width = xEnd - xStart
            height = yEnd - yStart
            triggerAreaImg = currImg[yStart:yStart + width,xStart:xStart + width]
            lastImg = lastImg[yStart:yStart + width,xStart:xStart + width]
            frameDelta = cv2.absdiff(triggerAreaImg, lastImg)
            self.lastImg = currImg
            if np.average(frameDelta) > 5:
                return True
        else:
            self.lastImg = currImg

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


    def view(self,v):
        """apply view transformation on vector array"""
        return np.dot(v - self.state.pivot, self.state.rotation) + self.state.pivot - self.state.translation


    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(self, out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(self, out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)


    def pointcloud(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s])
        else:
            proj = self.project(view(verts))

        if self.state.scale:
            proj *= 0.5**self.state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]


    def getPointCloudImage(self,colorImage,colorFrame,depthFrame,pointCloudImage):
        depth_colormap = np.asanyarray(
            self.colorizer.colorize(depthFrame).get_data())
        if self.state.color:
            mapped_frame, color_source = colorFrame, colorImage
        else:
            mapped_frame, color_source = depthFrame, depth_colormap
        points = self.pc.calculate(depthFrame)
        self.pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        now = time.time()
        self.pointCloudImage.fill(0)
        self.grid(pointCloudImage, (0, 0.5, 1), size=1, n=10)
        self.frustum(pointCloudImage, self.depth_intrinsics)
        self.axes(pointCloudImage, self.view([0, 0, 0]), self.state.rotation, size=0.1, thickness=1)

        if not self.state.scale or pointCloudImage.shape[:2] == (self.h, self.w):
            self.pointcloud(pointCloudImage, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            self.pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, pointCloudImage.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(pointCloudImage, tmp > 0, tmp)

        if any(self.state.mouse_btns):
            axes(pointCloudImage, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now
        return pointCloudImage


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
                motionDetected = self.detectMotion(color_image, self.lastImg)
                if motionDetected == True:
                    print("movement")

                # Remove background - Set pixels further than clipping_distance to grey
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                white_color = 255
                bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), white_color, color_image)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                self.pointCloudImage = self.getPointCloudImage(color_image ,color_frame ,aligned_depth_frame, self.pointCloudImage)
                self.display(color_image, bg_removed, depth_colormap, self.pointCloudImage)

                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.pipeline.stop()