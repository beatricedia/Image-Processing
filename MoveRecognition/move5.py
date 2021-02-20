import numpy as np
import cv2
from datetime import datetime

sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX
#TODO: Face Detection 1

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    # print("Distanta: ",  dist)
    return dist

cv2.namedWindow('frame')
cv2.namedWindow('dist')

#capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
cap = cv2.VideoCapture(0)

_, frame1 = cap.read()
_, frame2 = cap.read()

facecount = 0
nrOfSeconds = 0
dateFrame1 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3] 
while(True):
    _, frame3 = cap.read()
    # dateFrame1 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3] 
    rows, cols, _ = np.shape(frame3)
    cv2.imshow('dist', frame3)

    

    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)
    
    cv2.imshow('dist', mod)
    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
    dateFrame2 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3] 

    if stDev > sdThresh:
            print("Motion detected.. Do something!!!  ", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3], "  ", stDev)
            nrOfSeconds = 0
            dateFrame1 = dateFrame2
            #TODO: Face Detection 2
            
    else:
            print("Motion Stopped   ", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3],"  ", stDev)
            print("df1: ", dateFrame1)
            print("df2: ", dateFrame2)
            if dateFrame1.split(":")[2] != dateFrame2.split(":")[2]:
                nrOfSeconds += 1
                print("nr of s: ",nrOfSeconds)
            if nrOfSeconds == 5:
                print("Stop!!!")
                nrOfSeconds = 0
            dateFrame1 = dateFrame2
            

    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
