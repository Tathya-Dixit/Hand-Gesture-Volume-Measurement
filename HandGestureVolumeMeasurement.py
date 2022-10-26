import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
lst = {}
print(lst)

while True:
    success, img = cap.read()

    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = hands.process(imgrgb)
    # print(res.multi_hand_landmarks)
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lst[id] = cx,cy
                
    
                cv2.circle(img,(cx,cy),8,(0,0,0),cv2.FILLED)
                cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            diffx = lst[4][0]-lst[8][0]
            diffy = lst[4][1]-lst[8][1]
            diff = (diffx**2+diffy**2)**0.5
            if(diffx>300):
                perx = 100
            elif(diffx<0):
                perx=0
            else:
                perx = (diffx/300)*100
            
            if(diffy>300):
                pery = 100
            elif(diffy<0):
                pery=0
            else:
                pery = (diffy/300)*100
            
            if(diff>300):
                per = 100
            elif(diff<0):
                per=0
            else:
                per = (diff/300)*100

            cv2.rectangle(img,(w,170),(w-240,0),(255,255,255),-1)
            cv2.putText(img,"--DATA--",(w-170,20),cv2.FONT_HERSHEY_PLAIN,1.3,(0,0,0),2)
            cv2.putText(img,"X:"+str(int(perx))+"%",(w-70,160),cv2.FONT_HERSHEY_PLAIN,1.3,(0,0,0),2)
            cv2.rectangle(img,(w-10,130),(w-70,130-int(perx)),(255,0,0),-1)
            cv2.rectangle(img,(w-10,130),(w-70,30),(0,0,0),2)
            cv2.putText(img,"Y:"+str(int(pery))+"%",(w-150,160),cv2.FONT_HERSHEY_PLAIN,1.3,(0,0,0),2)
            cv2.rectangle(img,(w-80,130),(w-150,130-int(pery)),(0,0,255),-1)
            cv2.rectangle(img,(w-80,130),(w-150,30),(0,0,0),2)

            cv2.putText(img,"V:"+str(int(per))+"%",(w-230,160),cv2.FONT_HERSHEY_PLAIN,1.3,(0,0,0),2)
            cv2.rectangle(img,(w-160,130),(w-230,130-int(per)),(0,255,0),-1)
            cv2.rectangle(img,(w-160,130),(w-230,30),(0,0,0),2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

