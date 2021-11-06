THE WHOLE CODE

# IMPORT LIRARIES

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Input,LSTM,Dense,Flatten,Dropout
from tensorflow.keras.models import Model

# PREPARE INPUT DATA IN ARRAY FORM# 

# FOR DRIBBLE#

length=[]
j=np.zeros((230,100,66))
m=0
for filename in os.listdir(r"C:\Users\STEVE\Desktop\samples\dribble\dribble_lite"):
 cap = 
cv2.VideoCapture(os.path.join(r"C:\Users\STEVE\Desktop\samples\dribble\dribble_lite",filenam
e))
 len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 print(len)
 length.append(len)
 k=0
 with mp_pose.Pose(
 min_detection_confidence=0.5,
 min_tracking_confidence=0.5) as pose:
 while cap.isOpened():
 success, image = cap.read()
 #Flip the image horizontally for a later selfie-view display, and convert
 #the BGR image to RGB.#
 image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 #To improve performance, optionally mark the image as not writeable to
 #pass by reference.
 image.flags.writeable = False
 results = pose.process(image)
 #Draw the pose annotation on the image.
 image.flags.writeable = True
 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 mp_drawing.draw_landmarks(
 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 list=[]
 try:
 for i,l in enumerate(results.pose_landmarks.landmark):
 image_height, image_width, _ = image.shape
 cx,cy=int(l.x*image_width),int(l.y*image_height)
 list.append((int(cx),int(cy)))
 except:
 list.append(np.zeros((2,33))) 
 pp=np.reshape(list,(66,))
 
 j[m,k,:]=pp
 k=k+1 
 if k==length[m]-1:
 
 break
 if k==100:
 break 
 
 
 m=m+1
mm=0
length=[]
 
# FOR SITUP#

for filename in os.listdir(r"C:\Users\STEVE\Desktop\samples\situp\situp_lite"):
 cap = 
cv2.VideoCapture(os.path.join(r"C:\Users\STEVE\Desktop\samples\situp\situp_lite",filename))
 len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 print(len)
 length.append(len)
 k=0
 with mp_pose.Pose(
 min_detection_confidence=0.5,
 min_tracking_confidence=0.5) as pose:
 while cap.isOpened():
 success, image = cap.read()
 #Flip the image horizontally for a later selfie-view display, and convert
 #the BGR image to RGB.
 image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 #To improve performance, optionally mark the image as not writeable to
 #pass by reference.
 image.flags.writeable = False
 results = pose.process(image)
 #Draw the pose annotation on the image.
 image.flags.writeable = True
 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 mp_drawing.draw_landmarks(
 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 list=[]
 try:
 for i,l in enumerate(results.pose_landmarks.landmark):
 image_height, image_width, _ = image.shape
 cx,cy=int(l.x*image_width),int(l.y*image_height)
 list.append((int(cx),int(cy)))
 except:
 list.append(np.zeros((2,33))) 
 pp=np.reshape(list,(66,))
 
 j[m,k,:]=pp
 k=k+1 
 if k==length[mm]-1:
 
 break
 if k==100:
 break 
 
 print(len) 
 m=m+1
 mm=mm+1 

# FOR PULLUP

mm=0
length=[]
for filename in os.listdir(r"C:\Users\STEVE\Desktop\samples\pullup\pullup_lite"):
 cap = 
cv2.VideoCapture(os.path.join(r"C:\Users\STEVE\Desktop\samples\pullup\pullup_lite",filename)
)
 len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 print(len)
 length.append(len)
 k=0
 with mp_pose.Pose(
 min_detection_confidence=0.5,
 min_tracking_confidence=0.5) as pose:
 while cap.isOpened():
 success, image = cap.read()
 #Flip the image horizontally for a later selfie-view display, and convert
 #the BGR image to RGB.
 image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 #To improve performance, optionally mark the image as not writeable to
 #pass by reference.
 image.flags.writeable = False
 results = pose.process(image)
 #Draw the pose annotation on the image.
 image.flags.writeable = True
 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 mp_drawing.draw_landmarks(
 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 list=[]
 try:
 for i,l in enumerate(results.pose_landmarks.landmark):
 image_height, image_width, _ = image.shape
 cx,cy=int(l.x*image_width),int(l.y*image_height)
 list.append((int(cx),int(cy)))
 except:
 list.append(np.zeros((2,33))) 
 pp=np.reshape(list,(66,))
 
 j[m,k,:]=pp
 k=k+1 
 if k==length[mm]-1:
 
 break
 if k==100:
 break 
 
 
 m=m+1
 mm=mm+1 
j.shape
w4d=j.reshape(230,6600)
# SAVING THE INPUT ARRAY TO ACCESS IT LATER
np.savetxt('train_array_mediapipe2.txt', w4d)
w=np.loadtxt("train_array_mediapipe2.txt")
w.shape
w=w.reshape(230,100,66)
w.shape

# CREATING THE EXPECTED OUTPUT DATA FOR TRAINING#

oo=[[1,0,0]]*85#dribble
ooo=[[0,1,0]]*(152-85)#situp
oooo=[[0,0,1]]*(230-152)#pullup
ytrain=np.concatenate([np.array(oo),np.array(ooo),np.array(oooo)])
ytrain.shape
 
# PREPARING DATA AND THEN TRAINING

from sklearn.utils import shuffle
x_train,y_train = shuffle(w,ytrain)
print(x_train.shape)
print(y_train.shape)
i=Input(shape=x_train[0].shape)
x=LSTM(128)(i)
x=Dense(100,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(3,activation="softmax")(x)
model=Model(i,x)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
r=model.fit(x_train,y_train,epochs=12)

# DEFINING FUCTION FOR PREDICTION 

def action(pr):
 qq=0
 count=0
 for q in range(0,3):
 if pr[0][q]>qq:
 qq=pr[0][q]
 count=q
 
 if count==0:
 cc="dribbling"

 
 elif count==1:
 cc="situp"

 elif count==2:
 cc="pullups"

return cc 

# USING WEB CAM TO TEST OUR MODEL

#cap=cv2.VideoCapture(r"C:\Users\STEVE\Desktop\samples\pullup\pullup_lite\46_Pull_ups
_pullup_f_cm_np1_fr_bad_2.avi")
cap=cv2.VideoCapture(0)
if not cap.isOpened():
 raise IOError("cannot open video")
 
j=np.zeros((500,66))
k=0
with mp_pose.Pose(
 min_detection_confidence=0.5,
 min_tracking_confidence=0.5) as pose:
 while cap.isOpened():
 success, image = cap.read()
 #Flip the image horizontally for a later selfie-view display, and convert
 #the BGR image to RGB.
 image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 #To improve performance, optionally mark the image as not writeable to
 #pass by reference.
 image.flags.writeable = False
 results = pose.process(image)
 #Draw the pose annotation on the image.
 image.flags.writeable = True
 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 mp_drawing.draw_landmarks(
 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 list=[]
 try:
 for i,l in enumerate(results.pose_landmarks.landmark):
 image_height, image_width, _ = image.shape
 cx,cy=int(l.x*image_width),int(l.y*image_height)
 list.append((int(cx),int(cy)))
 except:
 list.append(np.zeros((2,33))) 
 pp=np.reshape(list,(66,))
 
 j[k,:]=pp
 k=k+1 
 print(k)
 if k<=101:
 cv2.putText(image,"ANALYSING...." , 
((200,15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0,255), 2)
 else: 
 test=np.expand_dims(j, axis=0)
 pred=model.predict(test[0:1,(k-100):k])
 countt=action(pred)
 cv2.putText(image,countt , ((250,15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 
255,0), 2)
 #print(k)
 #print(j[k])
 cv2.imshow('MediaPipe Hands', image)
 if cv2.waitKey(5) & 0xFF == 27:
 break
cap.release()
 
# TESTING OUR MODEL WITH SAMPLE CLIPS

cap=cv2.VideoCapture(r"C:\Users\STEVE\Desktop\samples\testing\10YearOldYouthBasket
ballStarBaller_dribble_f_cm_np1_fr_med_2.avi")
#cap=cv2.VideoCapture(0)
if not cap.isOpened():
 raise IOError("cannot open video")
 
j=np.zeros((100,66))
k=0
with mp_pose.Pose(
 min_detection_confidence=0.5,
 min_tracking_confidence=0.5) as pose:
 while cap.isOpened():
 success, image = cap.read()
 #Flip the image horizontally for a later selfie-view display, and convert
  the BGR image to RGB.
 image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 #to improve performance, optionally mark the image as not writeable to
 #pass by reference.
 image.flags.writeable = False
 results = pose.process(image)
 #Draw the pose annotation on the image.
 image.flags.writeable = True
 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 mp_drawing.draw_landmarks(
 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 list=[]
 try:
 for i,l in enumerate(results.pose_landmarks.landmark):
 image_height, image_width, _ = image.shape
 cx,cy=int(l.x*image_width),int(l.y*image_height)
 list.append((int(cx),int(cy)))
 except:
 list.append(np.zeros((2,33))) 
 pp=np.reshape(list,(66,))
 
 j[k,:]=pp
 k=k+1 
 print(k)
 if k<=70:
 cv2.putText(image,"ANALYSING...." , 
((200,15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0,255), 2)
 else: 
 test=np.expand_dims(j, axis=0)
 pred=model.predict(test[0:1,:100])
 countt=action(pred)
 cv2.putText(image,countt , ((250,15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 
255,0), 2)
 #print(k)
 #print(j[k])
 cv2.imshow('MediaPipe Hands', image)
 if cv2.waitKey(5) & 0xFF == 27:
 break
cap.release()
