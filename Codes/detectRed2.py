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

    for pic, contour in enumerate(cnts):
                area = cv2.contourArea(contour)
                if(area>300):
                        
                        x,y,w,h = cv2.boundingRect(contour)     
                        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    for c in cnts:
	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cstring = str(cX) + ", " + str(cY)
 
	# draw the contour and center of the shape on the image
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, cstring, (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    if(len(cnts) > 0):
            print("red detected")
            print('#', i)
            i+=1

    #make centroid and lines
    (h, w) = frame.shape[:2] #w:image-width and h:image-height        
    cv2.circle(frame, (w//2, h//2), 7, (0, 0, 255), -1)
    cv2.line(frame,(w//2, w),(w//2,0),(0,0,255),1)
    cv2.line(frame,(w, h//2),(0,h//2),(0,0,255),1)
    
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
