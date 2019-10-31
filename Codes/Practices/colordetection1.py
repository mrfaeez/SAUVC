# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())
 
# load the image
# image = cv2.imread(args["image"])

img = cv2.imread('color1.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# define the list of boundaries
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask = mask)
	condition = output.any()
	if bool(condition):
                print("success")
 
	# show the images
	cv2.imshow("images", np.hstack([img, output]))
	cv2.waitKey(0)
