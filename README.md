# Exercise-correctness-classifier-movenet-lightning

## A EXERCISE CORRECTNESS CLASSIFIER USING MOVENET LIGHTNING POSE ESTIMATION AND DEEP NEURAL NETWORKS

Movenet Lightning is used to generate keypoints that will then be fed to a custom deep neural network model.


There will be five exercises, each with its own neural network binary classifier which classifies the given exercise as good execution or bad.

The exercises included are the following:
  - Pushup
  - Lunge
  - Plank
  - Squat
  - Legraise

Movenet Lightning: https://tfhub.dev/google/movenet/singlepose/lightning/4

Movenet Lightning Demo Colab: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/movenet.ipynb#scrollTo=SYFdK-JHYhrv


## Suggested order of execution of .py files
1. image_extractor.py
2. blurred_filter.py
3. duplicate_filter.py
4. create_final_dataset.py
5. trial_keypointsget.py
6. training_nn_classifiers.py
7. camera_testing.py

## Folders
- datasets_sample: contains 5 sample images per class per exercise from the dataset of images used to generate keypoints
- keypoints: contains 5 .csv files of generated keypoints from the full dataset
- lightning: the movenet lightning pre-trained pose estimation model
- models_v14: version 14 of the classifier models trained using the data from keypoints folder 
