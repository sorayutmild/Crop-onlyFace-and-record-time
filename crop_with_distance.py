import datetime
import cv2
from time import sleep

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
found_face = False

def create_dataset(img): 
    cv2.imwrite("data/pic"+".jpg",img)
    datetime.datetime.now()
    file_time = open("time_collecting.txt","w")
    file_time.write(str(datetime.datetime.now()))

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        coords = [x,y,w,h]
    return img,coords

def detect(img,faceCascade):
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"EIEI")
    global found_face
    if len(coords) == 4 :
        create_dataset(img)
        found_face = True
    return img

cap = cv2.VideoCapture(0)
while(found_face == False):
    ret,frame = cap.read()
    frame = detect(frame,faceCascade)
    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()

