import numpy as np
import cv2

cap = cv2.VideoCapture('http://192.168.0.8:4747/mjpegfeed?640x480')

#Check whether user selected camera is opened successfully.

while cap.isOpened():

    #Capture frame-by-frame

    ret, frame = cap.read()

    # Display the resulting frame

    cv2.imshow('preview',frame)

    #Waits for a user input to quit the application

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

    # When everything done, release the capture

cap.release()

cv2.destroyAllWindows()