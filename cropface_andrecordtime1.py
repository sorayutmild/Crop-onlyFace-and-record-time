import datetime
import cv2
from time import sleep
import multiprocessing as mp


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
found_face = False
count_done = False
foundf = False

def count_up():
    #count up timer
    print("count start")
    count = 3
    for x in range(1,count+1):
        print(x, "second")
        sleep(1)
    count_done = True
    print("collect picture to data")

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

def found(img,faceCascade):
    print("found start")
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"EIEI")
    global found
    while(len(coords) == 4 ):
        foundf = True
    foundf = False
def check_and_create_dataset(img,count_done,foundf):
    if (count_done == True) and (foundf == True):
        create_dataset(img,id)
        found_face = True
    return True

def detect(img,faceCascade):
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"EIEI")
    global found_face
    global count_done
    global foundf
    if len(coords) == 4 :
        if(__name__ == '__main__'):
            p1 = mp.Process(target=count_up)
            p2 = mp.Process(target=check_and_create_dataset,args = (img,count_done,foundf))
            p3 = mp.Process(target=found,args = (img,faceCascade))
            p1.start()
            p2.start()
            p3.start()
            
    return img

cap = cv2.VideoCapture(0)
while(found_face == False):
    ret,frame = cap.read()
    frame = detect(frame,faceCascade)
    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()

