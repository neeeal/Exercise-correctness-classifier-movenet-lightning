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
- image_extractor.py
- blurred_filter.py
- duplicate_filter.py
- create_final_dataset.py
- trial_keypointsget.py
- training_nn_classifiers.py
- camera_testing.py
