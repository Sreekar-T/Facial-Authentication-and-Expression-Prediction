import os 
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model, model_from_json
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image
model = model_from_json(open("draft3.json", "r").read())
model.load_weights('draft3.h5')
#names = ["none","unknown","User"]
names = ["none","unknown","User","Sreekar"]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer1.yml')
cascadePath = "frontface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
face_haar_cascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
video_path="/content/1.jpg"
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)
while cap.isOpened():
    res,frame=cap.read()
    height, width , channel = frame.shape
    sub_img = frame[0:int(height/6),0:int(width)]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
    res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    lable_color = (10, 10, 255)
    lable = "Emotion Detection"
    lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
    textX = int((res.shape[1] - lable_dimension[0]) / 2)
    textY = int((res.shape[0] + lable_dimension[1]) / 2)
    cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for (x,y, w, h) in faces:
      cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
      roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
      roi_gray=cv2.resize(roi_gray,(48,48))
      image_pixels = img_to_array(roi_gray)
      image_pixels = np.expand_dims(image_pixels, axis = 0)
      image_pixels /= 255
      predictions = model.predict(image_pixels)
      max_index = np.argmax(predictions[0])
      emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
      emotion_prediction = emotion_detection[max_index]
      cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0,textY+22+5), FONT,0.7, lable_color,2)
      lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
      violation_text_dimension = cv2.getTextSize(lable_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
      violation_x_axis = int(res.shape[1]- violation_text_dimension[0])
      cv2.putText(res, lable_violation, (violation_x_axis,textY+22+5), FONT,0.7, lable_color,2)

      id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
      if (confidence < 100):
          id = names[id]
          confidence = "  {0}%".format(round(100 - confidence))
      else:
          id = "unknown"
          confidence = "  {0}%".format(round(100 - confidence))
        
      cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
      cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)


    frame[0:int(height/6),0:int(width)] =res
    cv2.imshow('Result',frame)
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows