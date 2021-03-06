# Deep Learning Project

Honey Bee health classification using CNN.
 
<br />

Repository Content

•	Project_ADL_32920.ipynb <br />
•	Images Folder - Images from results obtained.
 
## Abstract

Every third bite of food relies on pollination by bees. 
 Pollinators transfer pollen and seeds from one flower to another, 
 fertilizing the plant to it can grow and produce food.
Cross-pollination helps at least 30 percent of the world’s crops and 90 percent of our wild plants to thrive. Without bees to spread seeds, many plants—including food crops—would die off.(NRDC,2015)
 The main aim of this project is to track the health of Bees using the images of Bees in "The BeeImage Dataset: Annotated Honey Bee Images" dataset using the 
 convolutionalneural netwok. I have used transfer learning approach with Vgg-19 pretrained model to classify 
 health of Bees.



------------------------------------------------------------------------------



## Introduction

### The BeeImage Dataset
This dataset contains 5,100+ Bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.(Jenny,2019)

I have taken only Bee health in consideration and there are 6 classes related to Bee health namely 'hive being robbed', 'healthy', 'few varrao, hive beetles','ant problems', 'missing queen', 'Varroa, Small Hive Beetles'. I have split the dataset into train and test of 85% and 15% respectievely. Below are some images from dataset. The dataset can be downloaded from this link https://www.kaggle.com/jenny18/honey-bee-annotated-images/. 

<img src="Images/Images_dataset.png">


### Data preprocessing and Image augmentation


Since the dataset contains many fields irrelevent to health of Bees, therefore processed the data using Pandas Dataframe and created a  training labelled data of Bees Health. also all iamges are resized to fix size of 100 * 100 * 3.
Image augmentation is done to avoid overfitting of the model, ImageDataGenerator class from Keras is used to perform the augmentation. zoom,flip and shift techniques are used to perform augmenation.

### VGG-19


VGG-19 is a convolutional neural network that is 19 layers deep. I have used the V66-19 model by Keras pretrained on  millions of images of Imagenet dataset. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224*224. VGG19 is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer). 



### Libraries used:
Numpy,
Pandas,
OpenCV,
Matplotlib,
Keras,
scikit-learn


### Training of Model:
I have used the Transfer learning==> Feature extraction approach to train the model. I have used the VGG-19 model in Keras pretrained with 'Imagenet weights'. Since I have used Feature extraction approach to train the model, I have made changes only in last classification layer of model. Since the dataset contains 6 Health classes. I have added the output layer(softmax) which will output this 6 classes. The hyperparameters used for training are given below. I have got training accuracy 98% and testing accuracy of 92.4%



Hyperparameter | Epoch | Learning rate | Optimizer | Batch Size | loss
--- | --- | --- |--- |--- |---
Value | 50 | 0.0001  | Adam | 256 | categorical cross entropy 

<img src="Images/Graph.png">




## Evaluation of Model:

I have evaluated the model using precision,recall and F1-score metrics. To implement this I have used the classification_report function from scikit-learn library.

Precision - TP/(TP+FP) , where TP is true positive , FP is false positive ,FN is false negative  (Powers,2011)

Recall-  TP/(TP+FN) (Powers,2011)

F1-score - (2 * precision * recall) / (precision +recall) (Powers,2011)

<img src="Images/classification_report.png">

I have also plot the confusion matrix to evaluate the model. To implement this I have used confusion_matrix function from scikit-learn library.

<img src="Images/confusion_matrix.png">

## Results :

Since I have trained the model on 85% of images and remaining images are used for testing purpose, I have got the accuracy of 93% for the test images. Below are the some of the predictions of images from the testing data by the trained model.


<img src="Images/result_1.png">
<img src="Images/result_2.png">






## References

Jenny Yang (2019). The BeeImage Dataset: Annotated Honey Bee Images, https://www.kaggle.com/jenny18/honey-bee-annotated-images

NRDC (2015). Busy as a bee:Pollinators put food on the table, https://www.nrdc.org/sites/default/files/bee-deaths-FS.pdf

POWERS, D. (2011). EVALUATION: FROM PRECISION, RECALL AND F-MEASURE TO ROC,INFORMEDNESS, MARKEDNESS & CORRELATION. Journal of Machine LearningTechnologies ISSN: 2229-3981 & ISSN: 2229-399X, Volume 2, Issue 1, 37-63.






