import datetime
import cv2
from time import sleep
import multiprocessing as mp

time_count = 3
i = 0
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
    global i 
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"EIEI")
    global found_face
    global frame
    global cap
    global time_count
    while(len(coords) == 4 and i <=time_count):
        i += 1
        sleep(0.3)
        ret,frame = cap.read()
        img,coords = draw_boundary(frame,faceCascade,1.1,10,(0,0,255),"EIEI")
        print(i)
    if(i >= time_count):
        create_dataset(frame)
        found_face = True

    return img

cap = cv2.VideoCapture(0)
while(found_face == False):
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    frame = detect(frame,faceCascade)
cap.release()
cv2.destroyAllWindows()

