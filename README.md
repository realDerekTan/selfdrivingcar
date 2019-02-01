# selfdrivingcar
This is my first Self Driving Car project. In this project, I use end-to-end neural nets and Convolutional Neural Networks (CNNs) to drive a virtual car around AirSim.

# Navigation

**dataexploration.py**
This is the file where I mess around with the data and do some testing. In this file, the raw images taken from AirSim are processed and transported, where they are stored for later use.

**trainmodel.py**
This is the heart of the project. Most of the machine learning and training is done here. In this file, the processed images are used to train, test, and evaluate a regression model that improves over time. Uses Tensorflow 1.11.0 and Keras 2.2.4

**testmodel.py**
This is the file where we run the model on AirSim and test the results of the project.

# Additional Files

**my_model.h5**
To make life easier, I attached the already trained model that my computer produced. If you want to skip the training step and go straight to testing, use this file as the finished model.

**Cooking.py**
This is the file that processes all the raw images and converts them into data that we can use to train and evaluate our model.
