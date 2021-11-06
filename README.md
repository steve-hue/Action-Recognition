# PROPOSED METHOD
In order to create a model to learn the movements for an activity, we have given body posture as
input data into an LSTM layer .Hence learning the movements of a human over time. We have
taken the help of Mediapipe library. It is an open source library offering customizable cuttingedge ML solutions for live and stream media. The unique part is that it is built in fast MLinterface which gives faster computations and hence greater fps
One such module in the library is Pose classification. It predicts the positioning of 33 landmarks
on each frame, giving a skeleton outline to help estimate the pose.
The detector first locates the person/pose region-of-interest (ROI) within the frame. The tracker
subsequently predicts the pose landmarks within the ROI using the ROI-cropped frame as input. 

![image](https://user-images.githubusercontent.com/59787095/140625503-e88e05c2-f04b-4967-9c90-df6146c64466.png)

We have mentioned the working flow of our code below for better understanding

![image](https://user-images.githubusercontent.com/59787095/140625515-4ca2aa96-bb22-45b9-a424-2736551780b9.png)
