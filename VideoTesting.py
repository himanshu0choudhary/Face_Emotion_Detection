import os
import cv2 as cv
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')
face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv.VideoCapture(0)
while cap.isOpened():
    rate,img=cap.read()
    if not rate:
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        crop_img=img[y:y+w,x:x+h]
        crop_img=cv.resize(crop_img, (48, 48))
        img_pixels = image.img_to_array(crop_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv.putText(img, predicted_emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('img',img)
    if cv.waitKey(1) & 0xFF==ord('b'):
        break

cap.release()