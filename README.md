# Computer Vision Tasks

# Task - 1: Hand-Written Digits Classification with TensorFlow CNN
This project implements a Convolutional Neural Network (CNN) model from scratch using TensorFlow for image classification. The dataset used for this task consists of hand-written digits with three classes: 0, 1, and 2. The goal is to train a model to classify these digits accurately.

## Dataset
The dataset is organized in the following format:

```javascript
Hand_written_digits/
    0_digits/
        image1.jpg
        image2.jpg
        ...
    1_digits/
        image1.jpg
        image2.jpg
        ...
    2_digits/
        image1.jpg
        image2.jpg
        ...
```
The images in the dataset are grayscale and have a size of 100x100 pixels.

## Data Split
In the code, the dataset is split into training and validation sets. 80% of the data is used for training, and the remaining 20% is used for validation. TensorFlow's built-in functions are utilized to perform this split.

## Data Augmentation
Data augmentation is performed to enhance the dataset's diversity and robustness. However, it's important to note that some augmentations are not suitable for this dataset. For example, horizontal or vertical flips can change digit 2 into something else. Careful selection of augmentations is crucial.

## Model Architecture
The model architecture used for this task is a customizable CNN. You have the flexibility to modify the architecture based on your creativity and experimentation. Achieving the best possible accuracy is the primary objective.

## Training
The model is trained on the provided hand-written digits dataset. Training is conducted for 10 epochs by default. The task is evaluated based on the validation accuracy, so optimizing the model architecture is essential.

## Running the Code
Ensure the required libraries of python are properly installed. You can install it using pip:

Monitor the training process and analyze the validation accuracy to determine the model's performance.

## Results
After training the model for 10 epochs, the achieved accuracy on the validation set is reported. 


## Task 2: Fine-Tuning a Pre-trained Model
#### Choice of pre-trained model
#### Fine-tuning process and model architecture adjustments
#### Training and validation results
#### Accuracy Check

## Task 3: Image Retrieval System
#### Overview of the image retrieval system
#### Pre-trained model selection (e.g., VGG16)
#### Feature extraction and similarity metric (e.g., Euclidean distance)
#### Image Retreival
