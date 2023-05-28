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



'''
LOADING MODEL LOCALLY
'''

print('Loading Movenet Lightning pose estimation model')
module = tf.saved_model.load('D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\movenet\\lightning')
model = module.signatures['serving_default']
input_size = 192

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

exercise_path = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\dataset'
exercises = os.listdir(exercise_path)
exercises.sort()
##print(exercises)
for i,ex in enumerate(exercises):
##    if i == 0: continue
    class_path = os.path.join(exercise_path,ex)
    classes = os.listdir(class_path)
    classes.sort()
##    print(classes)
    for m,class_ in enumerate(classes):
        keypoints_arr = []
        images_path = os.path.join(class_path,class_)
        images_filenames = os.listdir(images_path)
        for n,img in enumerate(images_filenames):
            if n%500 == 0:
                print(f'Getting keypoints {ex} {class_} number {n}')
            image_path = os.path.join(images_path,img)
            # Load the input image.
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

            # Run model inference.
            keypoints_with_scores = movenet(input_image)
            keypoints_arr.append(keypoints_with_scores)
##            if n == 500: break
        if m == 0:
            zero_df = pd.DataFrame(data=np.array(keypoints_arr).reshape(np.array(keypoints_arr).shape[0],-1))
            zero_df[51] = np.zeros(shape=(np.array(keypoints_arr).shape[0],))
        else:
            one_df = pd.DataFrame(data=np.array(keypoints_arr).reshape(np.array(keypoints_arr).shape[0],-1))
            one_df[51] = np.ones(shape=(np.array(keypoints_arr).shape[0],))
            final_df = pd.concat([zero_df, one_df])
            
    keypoints_path = f'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\keypoints'
    if os.path.isdir(keypoints_path):
        pass
    else:
        os.mkdir(keypoints_path)
    filename = os.path.join(keypoints_path,f'{ex}.csv')
    final_df.to_csv(filename, sep=',', index=False, encoding='utf-8')
##    break
        
print('Finished Execution')

