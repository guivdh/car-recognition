import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

img_array = []

nbreVoiture = 0
nbreFrame = 0

cap = cv2.VideoCapture('video-voiture.avi')

fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame1 = cap.read()
ret2, frame2 = cap.read()

kernel = np.ones((5, 5), np.uint8)

# line
width, height = 800, 600
x1, y1 = 400, 0
x2, y2 = 400, 720
image = np.ones((height, width)) * 255
line_thickness = 2

position = (10, 50)

while cap.isOpened():
    cv2.putText(frame1, "Nombre de voiture : " + str(nbreVoiture), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                3)

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=5)

    grayImageBGRspace = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 30000:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        phrase = "Voiture "
        cv2.putText(frame1, phrase, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        if (x > 395 and x < 405):
            print("Ligne touchée")
            nbreVoiture += 1
            strg = "Nombre de voiture : " + str(nbreVoiture)
            cv2.putText(frame1, strg, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow("feed", frame1)

    smallFrame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
    smallGrayImageBGRspace = cv2.resize(grayImageBGRspace, (0, 0), fx=0.5, fy=0.5)

    cv2.line(smallFrame1, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

    vis = np.concatenate((smallFrame1, smallGrayImageBGRspace), axis=1)
    cv2.imshow('image', vis)
    frame1 = frame2
    ref, frame2 = cap.read()

    height, width, layers = vis.shape
    size = (width, height)

    nbreFrame += 1
    print(nbreFrame)
    img_array.append(vis)

    if nbreFrame == 294:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # time.sleep(1/fps)

# When everything done, release the capture
out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 50, size)
for i in range(len(img_array)):
    out.write(img_array[i])
cap.release()
cv2.destroyAllWindows()
