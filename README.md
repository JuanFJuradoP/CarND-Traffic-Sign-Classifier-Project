# Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used what I have learned about deep neural networks and convolutional neural networks to classify traffic signs. I trained and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Writeup
---

The writeup file include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as the brief description of each step-by-step. Images and snippets were include to demonstrate how the code works and show few examples. 

To meet specifications, the project will require submitting three files: 
* The Ipython notebook.
* The Ipython notebook exported as a HTML file.
* The writeup report. 

### Dataset Exploration

- *Dataset Summary* - The submission includes a basic summary of the data set.

- *Exploratory Visualization* - The submission includes an exploratory visualization on the dataset.

### Design and Test a Model Architecture

- *Preprocessing* - The submission describes the preprocessing techniques used and why these techniques were chosen.

- *Model Architecture* - The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

- *Model Training* - The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

- *Solution Approach* - The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

### Test a Model on New Images

- *Acquiring New Images* - The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

- *Performance on New Images* - The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

- *Model Certainty* - Softmax Probabilities - The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

Pipeline
---
The pipeline consist of:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Used Dependencies
---
To run the code succesfully you should have installed the following requirements.

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
