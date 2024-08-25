# CNN
CNN for image classification 

# DATASET 
+ The dataset is Animals-10 (from Kaggle), which you can find [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data).
+ The Animals-10 dataset from Kaggle contains around 28,000 images across 10 animal classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, and squirrel. It's ideal for image classification tasks and is widely used in computer vision projects and educational settings. The images are in JPG format and vary in resolution.

# SETTING
For optimizer and learning rate, there are 1 setting that I use:
++ SGD optimizer with different learning rates for each different epoch.

# TRAIN

# TEST 
If you want to train a model with common dataset and default parameters, you could run: 
> ** python train.py ** 

# EXPERIMENTS
I run the model in a machine named NVIDIA GeForce GTX 1650 Ti. The validation (showed in a picture below) was trained in 5th epoch (over 100 epochs) due to the device limitation. Thus, the accuracy could be improved more. 
![image](https://github.com/user-attachments/assets/674748b8-16b4-4ad7-8773-2291863a7127)

# DEPENDENCIES
+ Python 3.7 or above
+ Pytorch 2.1.2
+ Tensorboard 2.12.1


