import numpy as np
import cv2

cap = cv2.VideoCapture(2)

kernel = np.ones((5,5),np.uint8)

#Red Color
lower_color = np.array([161, 155, 84])
upper_color = np.array([179, 255, 255])

#Blue color
#lower_color = np.array([94, 80, 2])
#upper_color = np.array([126, 255, 255])

#Green color
#lower_color = np.array([25, 52, 72])
#upper_color = np.array([102, 255, 255])

#begin capture
i=0
while(True):
    ret, frame = cap.read()

    #Smooth the frame
    frame = cv2.GaussianBlur(frame,(11,11),0)

    #Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Mask to extract just the yellow pixels
    mask = cv2.inRange(hsv,lower_color,upper_color)

    #morphological opening
    mask = cv2.erode(mask,kernel,iterations=2)
    mask = cv2.dilate(mask,kernel,iterations=2)

    #morphological closing
    mask = cv2.dilate(mask,kernel,iterations=2)
    mask = cv2.erode(mask,kernel,iterations=2)

    #Detect contours from the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]

    if(len(cnts) > 0):
            print("red detected")
            print('#', i)
            i+=1

    #display the image
    cv2.imshow('frame',frame)
    #Mask image
    cv2.imshow('mask',mask)
    #Quit if user presses 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#Release the capture
cap.release()
cv2.destroyAllWindows()
