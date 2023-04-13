import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('/home/cloudvalley/Desktop/yolov5peoplecounterwin11/cctv.mp4')


tracker = Tracker()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    results=model(frame)
    list=[]
    j=0
    df = results.pandas().xyxy[0].rename(columns={'class':'clas'})
    for xmin, ymin, xmax, ymax, confidence, clas, name in df.itertuples(index=False):
        if name == "person":
            cv2.rectangle(results.ims[0], (int(xmin), int(ymin)), (int(xmax), int(ymax)),(0,255,0), 2)
            j+=1
            list.append([xmin,ymin,xmax,ymax])
    boxes_ids = tracker.update(list)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        #cv2.rectangle(results.ims[0], (int(xmin), int(ymin)), (int(xmax), int(ymax)),(255,0,255),2)
        #cv2.putText(results.ims[0],str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.putText(frame,"Person:{}".format(j), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        print("Yes,The images are loaded in yolo")
        print("Detected Person:", j)
    cv2.imshow('FRAME',frame)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
    
