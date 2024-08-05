import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow
#import tkinter


def convert(s):
    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += x

        # return string
    return new

def most_frequent(List):
    count = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > count):
            count = curr_frequency
            num = i

    return num

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")
offset= 20
imgSize = 300

labels= ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","YES","NO","LOVE YOU","NEXT","space"]
textbuff= [' ' for i in range(10)]
buffsize = 15
buffer= ['' for i in range(buffsize)]
counter= 0
temp= ' '
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset: x + w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize =cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw= False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw= False)

        buffer[counter] = labels[index]
        # print(buffer[counter])
        mfreq=most_frequent(buffer)
        if mfreq == 'NEXT':
            mfreq = ''
            buffer= ['' for i in range(buffsize)]
        if mfreq == 'space':
            mfreq = ' '
        if temp != mfreq:
            textbuff.pop(0)
            textbuff.append(mfreq)
            temp = mfreq
        for i in textbuff:
            print(i, end='')
        # print(counter)

        cv2.putText(imgOutput, mfreq,(x,y-40), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.putText(imgOutput, convert(textbuff), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    counter = (counter + 1) % buffsize
    cv2.imshow("image", imgOutput)
    key =cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
with open('output.txt','w') as file:
    file.write(convert(textbuff))
