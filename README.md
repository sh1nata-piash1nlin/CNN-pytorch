# CNN
CNN for image classification (single object) 

# DATASET 
The dataset is Animals-10 (from Kaggle), which you can find [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data). <br>
The Animals-10 dataset from Kaggle contains around 28,000 images across 10 animal classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, and squirrel. It's ideal for image classification tasks and is widely used in computer vision projects and educational settings. The images are in JPG format and vary in resolution.

# SETTING
For optimizer and learning rate, there are 1 setting that I use: <br>
+ SGD optimizer with different learning rates for each different epoch.

# TRAIN 
If you want to train a model with default parameters, you could run: 
> **python train.py** <br>

If you want to train a model with your preference parameters, like the batch size, you could run: <br>
> **python train.py -b 8**

# TEST 
If you want to test a model with default parameters, you could run: 
> **python test.py** <br>

If you want to test a model  in images_testing file with your preference parameters, you could run, for instance: <br>
> **python test.py -p2 image_testing/cow.jpg** <br>

Here is the visualization with default parameter after training 1st epoch: 

![image](https://github.com/user-attachments/assets/1e6be439-c9b2-4941-b247-89959ec0e10c)


# EXPERIMENTS
I run the model in a machine named <span style="background-color: red;">NVIDIA GeForce GTX 1650 Ti</span>. The validation (showed in a picture below) was trained in 5th epoch (over 100 epochs) due to the device limitation. Thus, the accuracy could be improved more. 

![image](https://github.com/user-attachments/assets/674748b8-16b4-4ad7-8773-2291863a7127)

Here is the visualization in confusion matrix after training 1st epoch: 

![image](https://github.com/user-attachments/assets/7c9c757f-879c-433f-a67d-9a928972d900)


# DEPENDENCIES
+ Python 3.7 or above
+ Pytorch 2.1.2
+ Tensorboard 2.12.1


