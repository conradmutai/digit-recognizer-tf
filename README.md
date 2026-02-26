# Digit Recognizer - Kaggle Competition (Validation Accuracy of 99.52%)
## Introduction
Kaggle has a set of beginner competition that challenge users to explore different deep learning networks, and this one specifically tackled the usage of Convolutional Neural Networks. After spending time reading and practicing through Understanding Deep Learning and took the time to use the skills learned and applied it into this space, in addition there was periods that required learning experiences to create a convolutional neural network with TensorFlow rather than the standard PyTorch that was learned with the textbook to widen my understanding of the different libraries.

## Performance of the Model
The model achieved a validation accuracy of **99.52%** which placed my model within approximately the **Top 10%** of competitors in the leaderboard, on the second attempt. There is still room for improvement in the model which will be noted and used later to learn and progress towards a complete model and hopefully a 99.6%+ model accuracy going forward, but as of February 25th, 2026 this is where the model stands.

## Data Generation
Data was provided by Kaggle for this competition which contained a test and training data set which was utilised to train our model, however, in order to prevent factors like model memorization (the process where the model memorizes patterns though a training set and fails ot generalize) the training data was split into a validation and training set. The validation set will be used to help the model learn to generalize well.

In addition to generate more training data than was available to help increase the model accuracy, `ImageDataGenerator` from `tensorflow.keras.preprocessing.image` was utilised, creating variations of images by doing transformation like a **rotation range** and **zoom range** of 10%, and **width** and **height shift ranges** of 0.1.

The reason this is so vital and helps with improving the model accuracy, is that it provides variation to the model challenging the model to recognize new patterns rather than memorizing the same one, which as stated before helps the model accuracy due to the fact that it prevents generalization, and this is quite a common thing that is done in the vision AI projects and more.

## Optimizers and Annealers
### Optimizers
Optimizers are utilized to prevent the model instability, and allowing smoother learning for the model, allowing for thus better performance and prediction, in addition to eliminating the model bias, and most importantly allowing the model to bypass its initial local minima to continue reducing loss, and increasing accuracy of the model. 

This was successfully carried out by the `Adam` optimizer, a well known optimizer that allows for the mean and the variance to be regulated by two decay rates, a learning rate, and a small positive constant to prevent these from reaching 0. 

This was implemented in our model allowing for better results in the validation and training accuracy which pre-optimizer, was unstable per each epoch, and was also was poor performance wise, having early stages of stagnation in the loss and accuracy, which was seen in some graphs, through the training and modification phase of this model.

### Annealers
Annealers are important to pushing the models performance early as it calls for the model to change the learning rate based on several factors, such as the **patience** which is set to 5 and a **factor** of 0.5, while focusing on the *val_accuracy* as the main focus of the annealer, all while keeping a minimum learning rate of 0.00001 to prevent the model from learning at a rate which is too slow.

#### Patience
Patience is the method of measuring when the model should adjust the learning rate based on the change of either accuracy/loss of the validation/training set. With a value set to 5 like the model has, it means that the learning rate will only adjust after **5 epochs** has passed, it is better to choose a number like this as it prevents too many spikes on the learning rate, allowing the program to take more time learn at set rates and see its effectiveness before changing it.

#### Factor
Factor is a value that mulitplies the learning rate in order to reduce it whenever we have reached the end of the patience period (5 epochs in this case), choosing 0.5 provides a sort of middle ground that allows the learning rate to gradually change without causing stagnation in the learning through a factor which is too large (~0.9) and preventing exploding gradients through a factor which is too small (~0.01). 

## About Model
The model was trained on a data set provided by the Kaggle competition with a network consisting of three `Conv2D` (2-dimensional convolution) layers followed by `ReLU` activation, with filters of 32, 64, 128 and kernel sizes of 3x3 as a final result of tweaking the layers, allowing be to achieve high validation accuracy. 

It also implemented methods such as 