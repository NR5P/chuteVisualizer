import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2

VID_HEIGHT = 480
VID_WIDTH = 640


# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.depth, VID_WIDTH, VID_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, VID_WIDTH, VID_HEIGHT, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# remove objects over certain distance
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object, align images
align_to = rs.stream.color
align = rs.align(align_to)

# create buttons on right of image
buttonImg = np.zeros((VID_HEIGHT, 200, 3), np.uint8)
half = int(VID_HEIGHT/2)
buttonImg[...] = 255
buttonImg[half-5:half+5,:,:] = 0
cv2.putText(buttonImg, 'Capture',(25,int(half/2)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
cv2.putText(buttonImg, 'Area',(25,int(half/2+30)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
cv2.putText(buttonImg, 'Trigger',(25,int(half*1.5)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
cv2.putText(buttonImg, 'Area',(25,int(half*1.5+30)),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
cv2.setMouseCallback('Control',process_click)

def process_click(event, x, y, flags, params):
    pass

def handleButtonPress(button):
    pass

def preProcessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured_img = cv2.GaussianBlur(gray_img,(5,5),1)
    canny_img = cv2.Canny(blured_img,200,200)
    kernel = np.ones((2,2))
    dilated = cv2.dilate(canny_img,kernel,iterations=2)
    preproc_img = cv2.erode(dilated,kernel,iterations=1)

    return preproc_img


def display(image):
    imgWithButtons = np.hstack((image,buttonImg))
    cv2.namedWindow("chute visializer", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("chute visializer", imgWithButtons)



try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

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
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), white_color, color_image)

        threshold_img = preProcessing(bg_removed)
        contours, hierarchy = cv2.findContours(threshold_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #cv2.imshow('threshold', threshold_img)

        display(color_image)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.drawContours(color_image, contours, -1, (0,255,0), 3)
        images = np.hstack((color_image, depth_colormap))
        #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

