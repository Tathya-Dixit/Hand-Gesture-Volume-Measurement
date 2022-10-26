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
lst = np.zeros(21)

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
                lst[id] = cx
                
    
                cv2.circle(img,(cx,cy),8,(0,0,0),cv2.FILLED)
                cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            diff = lst[4]-lst[8]
            if(diff>250):
                per = 100
            elif(diff<0):
                per=0
            else:
                per = (diff/250)*100
            cv2.putText(img,str(int(per))+"%",(200,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.rectangle(img,(w-30,130),(w-40,30),(255,0,0),2)
            cv2.rectangle(img,(w-30,130),(w-40,130-int(per)),(255,0,0),-1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

