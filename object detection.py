import cv2
import numpy as np
import math

#tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture('7.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=100)#高斯背景消除
kernelOp = np.ones((5,5),np.uint8) 
kernelCl = np.ones((5,5),np.uint8) 
_,frame = cap.read()
#print(frame.shape)
#frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
#bbox = cv2.selectROI(frame)
#print(bbox)


cv2.destroyAllWindows()
idx=0
cars=np.zeros((0,5),int)
#tracker = cv2.TrackerKCF_create()
pid = 1
#for video7
roi_x=200
roi_y=330
roi_w=900
roi_h=684
#for video 8
'''roi_x=324
roi_y=233
roi_w=820
roi_h=531'''

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1280,720))

while True:
    ret,frame = cap.read()
    if ret == False: break
    frame = cv2.resize(frame,(1280,720))
    roi = frame[roi_y:roi_h,roi_x:roi_w]
    #cv2.rectangle(frame, (roi_x,roi_y), (roi_w, roi_h), (0, 0, 255), 2)#ROI範圍
    #cv2.line(roi,(0,int(roi.shape[0]*0.66)),(roi.shape[1],int(roi.shape[0]*0.66)),(255,255,255),5,2)#最大判定範圍
    #cv2.line(roi,(0,200),(600,200),(255,255,0),5,2)
    
    mask = object_detector.apply(roi)
    mask2 = object_detector.apply(roi)
    _, mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)#移除影子 大於254變為255 其餘為0
    
    mask = cv2.dilate(mask, None, iterations=4) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelOp) # 找到所有細小孔洞 閉運算
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelCl)# 移除噪聲資料  開運算
    #mask = cv2.dilate(mask, None, iterations=3) 
    
    
   
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>700:
            x,y,w,h = cv2.boundingRect(cnt) 
            print(x,y,w,h)
            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            new_car = True
            if w<40 or h<70: new_car=False      
            #更新位置(追蹤)
            min_dist=10000
            temp = -1
            bbox_n=0  
            for car in cars:
                #目前輪廓與舊有的物件相近(表示並非新的物件)
                if  abs(x-car[1])<w and abs(y-car[2])<h:
                    new_car = False
                    #以最近距離來確定要更新的物件
                    dist = math.sqrt((x-car[1])**2+(y-car[2])**2)
                    if min_dist>dist:
                        min_dist = dist
                        temp = car[0]
                #若物件超出追蹤範圍則刪除
                if(car[2]>=int(roi.shape[0]*0.66)):
                    #print("delete",car[0],car[1],car[2],car[3],car[4],len(cars)-1)
                    cars = np.delete(cars,(bbox_n),axis=0)
                bbox_n+=1
            #用目前輪廓更新舊有的位置
            for car in cars:
                if car[0] == temp:
                    car[1] = x
                    car[2] = y
                    car[3] = x+w
                    car[4] = y+h
                    cv2.putText(roi, str(car[0]), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                
            #判斷新的目標    
            if new_car == True and y<int(roi.shape[0]*0.33): #220
                cars = np.append(cars,np.array([[pid,x,y,x+w,y+h]]),axis=0)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #print('new',pid,x,y,x+w,y+h,len(cars))
                cv2.putText(roi, str(pid), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);
                pid+=1
            
                 
    #cv2.line(roi,(0,int(roi.shape[0]*0.66)),(roi.shape[1],int(roi.shape[0]*0.66)),(255,255,255),5,2)#最大判定範圍
    #cv2.line(roi,(0,int(roi.shape[0]*0.33)),(roi.shape[1],int(roi.shape[0]*0.33)),(255,255,255),5,2)#最大判定範圍        
    #   cv2.rectangle(frame, (roi_x,roi_y), (roi_w, roi_h), (0, 0, 255), 2)
    #out.write(frame)
    cv2.imshow('roi',mask2)
    cv2.imshow('mask',mask)
    cv2.imshow('video',frame)
    key = cv2.waitKey(10)
    if key == 32: cv2.waitKey(0)
    if key == 27: break
#out.release()
cv2.destroyAllWindows()