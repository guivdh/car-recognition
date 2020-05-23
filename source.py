import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

nbreVoiture = 0

cap = cv2.VideoCapture('video-voiture.avi')

fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame1 = cap.read()
ret2, frame2 = cap.read()

kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    grayImageBGRspace = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
       (x, y, w, h) = cv2.boundingRect(contour)

       if cv2.contourArea(contour) < 1000:
            continue
       cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
       phrase = "Voiture " + str(nbreVoiture)
       cv2.putText(frame1, phrase, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    #cv2.imshow("feed", frame1)
    vis = np.concatenate((frame1, grayImageBGRspace), axis=1)
    cv2.imshow('image', vis)
    frame1 = frame2
    ref, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    nbreVoiture+=1
    time.sleep(1/fps)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
