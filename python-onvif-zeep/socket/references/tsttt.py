import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


filedir = "/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb"
filename = "rgb_1189.jpg"
outputdir = "/home/sdt/Workspace/hojun/beat/liquid_amount_analysis/amount_prediction_data_output"

##https://stackoverflow.com/questions/61641204/background-removal-using-opencv-python
img = cv2.imread(f'{filedir}/{filename}')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = 70   
im_b = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(image = im_b, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse= True)
mask = np.ones(img.shape[:2], np.uint8)
mask.fill(255)
cv2.drawContours(mask, contours, contourIdx =0 , color =0, thickness = -1)
new_img = cv2.add(im_b, mask)
cv2.imwrite('masked.jpg',new_img)
new_img
plt.imshow(new_img)
plt.show()
img_gray.shape
cv2.imshow('masked.jpg', img_gray)
cv2.waitKey()

# Python program to explain cv2.imshow() method

# importing cv2
import cv2
filedir = "/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb"
filename = "rgb_1189.jpg"
outputdir = "/home/sdt/Workspace/hojun/beat/liquid_amount_analysis/amount_prediction_data_output"

##https://stackoverflow.com/questions/61641204/background-removal-using-opencv-python
# img = cv2.imread(f'{filedir}/{filename}')
# path
# path = r'C:\Users\Rajnish\Desktop\geeksforgeeks.png'

# Reading an image in default mode
image = cv2.imread("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_1189.jpg")

# Window name in which image is displayed
window_name = 'image'

# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()