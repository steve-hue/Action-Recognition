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



# Importing Libraries
We used various libraries having different functions and serving their own purpose to have an
accurate output.
OpenCV is mainly aimed at real-time computer vision applications. It mainly focuses on image
processing, video capture and analysis including features like face detection and object detection.
MediaPipe is a framework for building multimodal (for example video, audio, any time series
data), cross platform applied ML pipelines.
NumPy is a Python library used for working with arrays and has functions for working in domain
of linear algebra, Fourier transform and matrices.
Matplotlib is a comprehensive library for creating static, animated and interactive visualization
in Python.
OS provides function for interacting with the operating system. The OS and os.path modules
include many functions to interact with eh file system.
Tensorflow is an open-source library for machine learning which can be used across a range of
tasks but has a particular focus in training and interference for deep neural networks.

# i.Getting pose landmarks
As mentioned earlier, we need an array of pose landmarks as input data. To carry on with this
step we first save all our training videos of each activity in separate folders. Each training video
contains short (3-4secods) clips of person performing the activity. With the help of OS library we
access the training folder to perform the computations on each clip sequentially. Using opencv,
each frame of a clip is read and fed into the pose classification module mentioned above, giving
an array of pose landmarks which is appended into an empty list. This process is looped over
each clip in the entire folder until the points of the final clip is saved.
The above process is followed for all the training activity folders. In a nutshell, this step is to
convert the visual 3D activity into a numerical arrays that can be read by computers to perform
further task

# ii. Prepare Input-Output data:
Before we give our input data for training, it requires some cleaning and structuring. Since we
are going to later feed it into LSTM architecture, the arrays are reshaped with the help of Numpy
to make it compatible. The input array is reshaped into a 3D array as such
X_train = np.array((m,t,p)) , where m : No. of total clips(all activities included)
 t : No. of frames per clip
 p : The pose landmarks of each frame in list form
Here we take in only the first 100 frames of each clip. In case the clip is less than 100 frames,
rest of the array is padded with zeros
Since LSTM is a supervised learning algorithm, we create the necessary output data to compute
loss and optimizing it on training.
The output is of the following shape:
Y_train = np.array((m,o)) , where m : No. of total clips(all activities included)
 o : One-hot encoding of all the activities 
 
# iii. Model Architecture:
After our data has been prepared, we built a simple LSTM architecture to train our model on.
This step is really important as our models learning performance depends totally on the
algorithm and architecture chosen. We went with the following:

![image](https://user-images.githubusercontent.com/59787095/140625558-315dfe33-38ee-419d-bf43-d4d658db1a1a.png)


“Binary-cross entropy” loss was computed for each iteration and “adams” optimizer formulas
were used to update the weights. The whole layout was constructed with the help of Tensorflow
library

# iv. Training &Testing
The above architecture has been trained for 12 epochs. We had to keep in mind not to
unnecessarily over fit the data. Also, given the fact that the optimizer chosen is “adams”, hence
the optimizing of weights takes place at a lot quicker speed. These reasons led us to finalise on
12 epochs
The model was later tested successfully giving the expected results. For testing, the pose
landmarks is obtained and passed through the trained weights frame by frame, giving a
probability of the activities as output

# RESULTS
The model was successfully trained with 94% accuracy giving a rendered video with output at
20fps.We can also enable it using webcam. It takes in the past 100 frames of real time and
predicts an output at every frame. 

![image](https://user-images.githubusercontent.com/59787095/140625594-a56463d8-c10a-430e-81fa-5e2f43ea6224.png)
![image](https://user-images.githubusercontent.com/59787095/140625603-e066d6a7-ec90-4f1e-b6cd-1df7129390ad.png)

Our aim was to get the model to learn the behavioural gesture for each activity. That could be
done by estimating pose with the help of mediapipe library giving 33 landmarks on one’s body.
Our intentions to correctly study the movement of these pose landmarks over a series of 100
frames was achieved with the help of LSTM algorithm. The architecture mentioned in the above
section was able to form an understanding of the body movements and classify various activities
successfully.
Our model has shown 12% increase in accuracy while achieving much better fps when compared
to the rest.

![image](https://user-images.githubusercontent.com/59787095/140625625-c3389a69-bf9c-4f4e-9781-7cf71dd7ea65.png)


# Conclusion

The details mentioned in the proposed method was our method of doing this project. This same
project can be done in other ways too. This particular project/code can be analyzed, explored and
improved and can be used in real world application. Human Activity Recognition is currently
being used in several fields for different purposes and these fields is not just restricted to
technology field. Apart from technology field, it is being used in medicine field, psychology
field, commercial purpose and many other field. This indicates the scope of this algorithm is so
powerful that it still has potential to increase its usage rate. We believe that machine learning
engineers should utilize this algorithm for their project and moreover, even researchers can dig
deep into this.
We achieved 94% accuracy with LSTM training, media pipe library and training 100 frames per
second. We compared this model with other model, for example without media pipe or using
RNN training. After comparing we discovered that LSTM with media pipe achieved best results.



