import cv2
import os
from time import sleep
import numpy as np
import datetime

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_id = 0
found_face = False

def create_dataset(img_without_rec, id, img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img_without_rec)
    datetime.datetime.now()
    file_time = open("time_collecting.txt","w")
    file_time.write(str(datetime.datetime.now()))

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    img_without_rec = img.copy()
    #overlay = img.copy()
	#output = img.copy()
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color,2)
        cv2.putText(img, text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
        global area
        area = w*h
        print(area)
        coords = [x,y,w,h]
    return img,coords,img_without_rec

def detect(img,faceCascade,img_id):
    img,coords, img_without_rec = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"EIEI")
    global found_face
    if len(coords) == 4:
        #img(y:y+h,x:x+w)
        id = 1
        #crop_face = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        if area >=70000 :
            create_dataset(img_without_rec,id,img_id)
            found_face = True
       # os.startfile('camerasound.mp3')
    return img

cap = cv2.VideoCapture(0)
while(found_face == False):
    ret,frame = cap.read()
    frame = detect(frame,faceCascade,img_id)
    cv2.imshow('frame',frame)
    img_id += 1

cap.release()
cv2.destroyAllWindows()
