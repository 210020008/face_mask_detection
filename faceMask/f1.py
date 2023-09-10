import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('f3.jpg')


haar_data = cv2.CascadeClassifier('f2.xml')

# while True:
#     faces = haar_data.detectMultiScale(img)
#     for x,y,w,h in faces :
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
#     cv2.imshow('result',img)
#     if cv2.waitKey(2)==27 :
#         break
# cv2.destroyAllWindows()    

capture = cv2.VideoCapture(0)
data_without_mask = []
while True:
    flag , img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces :
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h , x:x+w, : ]
            face = cv2.resize(face,(50,50))
            print(len(data_without_mask))
            if(len(data_without_mask)<400):
              data_without_mask.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27 or len(data_without_mask)>=200:
            break

capture.release()
cv2.destroyAllWindows()            

np.save('with_mask.npy',data_without_mask)


