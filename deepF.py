#!pip install deepface

import cv2
import time
from numba import jit
from deepface import DeepFace
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("E:\\Anurag\\My_Programs\\deepface\\CV_tasks\\Video1.mp4")
face_cascade = cv2.CascadeClassifier('C:\\Users\\Anurag\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
pTime = 0   # for FPS


while True:
    
    success, img = cap.read()# read the video 

    if success == True:
        img = cv2.resize(img, None, None, fx=0.5, fy=0.5)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        #cv2.imshow("image", img)
        
        predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False) 
        
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        #rectangle for faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        cv2.putText(img, predictions['dominant_emotion'] ,(30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),1 )
        cv2.imshow("Imagefinal",cv2.putText(img, f'FPS: {int(fps)}',(30,50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),1) )
            
        k = cv2.waitKey(1) & 0xff
        if k==27:
            break
cap.release()
cv2.destroyAllWindows()


    
  #cv2.waitKey(28)