#yolo tested on images and also input from webcam
#CarCounter
import numpy as np
import torch
import cvzone
from ultralytics import YOLO
import cv2 as cv
import math
from sort import *
# child1=cv.imread('child.jpg')
# child1=cv.resize(child1,(2000,4000))

# model=YOLO('yolov8l.pt') # different weights as per need yolo8n,yolo8l,yolo8m
# )
# results=model('child2.jpg',show=True) #first weights will be downloaded
# cv.waitKey(0)
# cv.destroyAllWindows(
classNames = [
    "Person",
    "Bicycle",
    "Car",
    "Motorbike",
    "Aeroplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic Light",
    "Fire Hydrant",
    "Stop Sign",
    "Parking Meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Giraffe",
    "Backpack",
    "Umbrella",
    "Handbag",
    "Tie",
    "Suitcase",
    "Frisbee",
    "Skis",
    "Snowboard",
    "Sports Ball",
    "Kite",
    "Baseball Bat",
    "Baseball Glove",
    "Skateboard",
    "Surfboard",
    "Tennis Racket",
    "Bottle",
    "Wine Glass",
    "Cup",
    "Fork",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Sandwich",
    "Orange",
    "Broccoli",
    "Carrot",
    "Hot Dog",
    "Pizza",
    "Donut",
    "Cake",
    "Chair",
    "Sofa",
    "Potted Plant",
    "Bed",
    "Dining Table",
    "Toilet",
    "TV",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell Phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Refrigerator",
    "Book",
    "Clock",
    "Vase",
    "Scissors",
    "Teddy Bear",
    "Hair Drier",
    "Toothbrush"
]
totalCount=[]
device1 = "cuda" if torch.cuda.is_available() else "cpu"
print(device1)
cap=cv.VideoCapture('cars3.mp4')
model=YOLO('yolov8l.pt')
#using masking to increase accuracy and computation
mask=cv.imread('mask.png')
#detect car,motorbike and truck
limits=[755,828,984,900]
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)#if an id lost how many frames to wait once lost
while(True):
    isTrue,img=cap.read()
    imgRegion = cv.bitwise_and(img, mask)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5)) #default array
    for  r in results: #returns pbjects in the frame
        boxes=r.boxes #boxes is array of xyxy values of each objects
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(255,0,200),3)
            w,h=x2-x1,y2-y1
            #confidence of each objecy identification
            conf=math.ceil((box.conf[0]*100))/100
            # cvzone.putTextRect(img,f'{conf}',(x1,y1-20))
            # #class names identtified in the array as per their ids
            clas=int(box.cls[0])
            currentClass = classNames[clas]
            if currentClass=='Car' or currentClass=='Truck' or currentClass=='Bus' or currentClass=='Motorbike' and conf>0.3:
                #cvzone.putTextRect(img,f' ""{currentClass}{conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(0, 255, 0))
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))#stacking old detections with new detections
    cv.line(img,(200,671),(1037,671),(0,255,0),5)
    resultsTracker=tracker.update(detections)
    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        print(results)
        cvzone.cornerRect(img, (x1, y1, w, h), l=10,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f' ""{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx,cy=x1+w//2,y1+h//2
        cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
        if 600<cx<1037 and 651-100<cy<691+100 : #appending ids
            if totalCount.count(Id)==0:#check for unique ids in the list if id cnt=0 it would be counted as unique by the counter
                totalCount.append(Id)
                print(len(totalCount))
    cvzone.putTextRect(img, f' ""{int(len(totalCount))}', (50, 50), scale=2, thickness=3, offset=10)
    cv.imshow('video',img)
    if cv.waitKey(10)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()
