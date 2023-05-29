import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import os
import pandas as pd

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

use='squat'

print('Loading Movenet Lightning pose estimation model')
module = tf.saved_model.load('D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\movenet\\lightning')
model = module.signatures['serving_default']
input_size = 192

print('Loading NN classifier model')
model_path=f'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\models\\good_models\\models_v14\\{use}.h5'
classifier = tf.keras.models.load_model(model_path)
classifier.summary()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

# Open the device at the ID 0
# Use the camera ID based on
# /dev/videoID needed
vid_path = f'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\test_set\\{use}\\good.mp4'

cap = cv2.VideoCapture(vid_path)
##cap = cv2.VideoCapture(1)

#Check if camera was opened correctly
if not (cap.isOpened()):
    print("Could not open video device")
else:
    print("Camera opened")

cam_size = 360
#Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size)

# Capture frame-by-frame
scores = []
remarks = ['Very Bad','Bad','Good','Very Good']
i=0
while(True):
    ret, frame = cap.read()

    # Display the resulting frame
    
    cv2.imshow("preview",frame)
    
    cv2.imwrite("outputImage.jpg", frame)
    # Getting only every 5th frame
##    if i >= 5:
    ##    image = tf.image.decode_jpeg(frame)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoints_with_scores = movenet(input_image)
    output = classifier(keypoints_with_scores.reshape(1,51))

    # Evaluate scores every 15 frames
    score = np.array(output)[0][0]
    scores.append(score)
    if len(scores)==30:
        ave_score = sum(scores)/len(scores)
##        if ave_score < 0.25:
##            remark = remarks[0]
##        elif ave_score >= 0.25 and ave_score < 0.50:
##            remark = remarks[1]
##        elif ave_score >= 0.50 and ave_score < 0.75:
##            remark = remarks[2]
##        else:
##            remark = remarks[3]
        if ave_score < 0.60:
            remark = remarks[1]
        else:
            remark = remarks[2]
        confidence = ave_score*100    
        print(f'Correctness = {confidence:.4f}          Remark = {remark}')
        scores=[]
    i=0
##    else:
##        i+=1
##        pass

    #Waits for a user input to quit the application 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if ret == False:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
