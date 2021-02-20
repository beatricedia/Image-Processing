import numpy as np
import cv2
from datetime import datetime
import face_recognition

obama_image = face_recognition.load_image_file("./img/unknown/mama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file('./img/unknown/diana.jpg')
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "carmen",
    "diana"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

sdThresh = 25
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
    small_frame = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame1, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('frame', frame1)
    
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
            cv2.putText(frame2, "Motion - {}".format("Moving"), (90, 100), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

            nrOfSeconds = 0
            dateFrame1 = dateFrame2
            
    else:
            print("Motion Stopped   ", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3],"  ", stDev)
            cv2.putText(frame2, "Motion - {}".format("Stopped moving"), (90, 100), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

            print("df1: ", dateFrame1)
            print("df2: ", dateFrame2)
            if dateFrame1.split(":")[2] != dateFrame2.split(":")[2]:
                nrOfSeconds += 1
                print("nr of s: ",nrOfSeconds)
            if nrOfSeconds == 5:
                print("Stop!!!")
                nrOfSeconds = 0
            dateFrame1 = dateFrame2
            
    
   

    # cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
