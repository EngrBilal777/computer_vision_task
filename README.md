# Computer Vision Tasks

## Task - 1: Hand-Written Digits Classification with TensorFlow CNN
This project implements a Convolutional Neural Network (CNN) model from scratch using TensorFlow for image classification. The dataset used for this task consists of hand-written digits with three classes: 0, 1, and 2. The goal is to train a model to classify these digits accurately.

### Dataset
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

### Data Split
In the code, the dataset is split into training and validation sets. 80% of the data is used for training, and the remaining 20% is used for validation. TensorFlow's built-in functions are utilized to perform this split.

### Data Augmentation
Data augmentation is performed to enhance the dataset's diversity and robustness. However, it's important to note that some augmentations are not suitable for this dataset. For example, horizontal or vertical flips can change digit 2 into something else. Careful selection of augmentations is crucial.

### Model Architecture
The model architecture used for this task is a customizable CNN. You have the flexibility to modify the architecture based on your creativity and experimentation. Achieving the best possible accuracy is the primary objective.

### Training
The model is trained on the provided hand-written digits dataset. Training is conducted for 10 epochs by default. The task is evaluated based on the validation accuracy, so optimizing the model architecture is essential.

### Running the Code
Ensure the required libraries of python are properly installed. You can install it using pip:

Monitor the training process and analyze the validation accuracy to determine the model's performance.

### Results
After training the model for 10 epochs, the achieved accuracy on the validation set is reported. 


## Task 2: Fine-tuning a Pre-trained Model for Image Classification
In this task,fine tuning of the pretrained model have focused on transfer learning, making this task an opportunity to solidify your understanding of transfer learning and how to fine-tune a pre-trained model for your custom dataset. The main objective is to observe the accuracy of the fine-tuned model and compare it with the accuracy achieved in Task 1.

### Task Overview
In Task 1, I have implemented a basic Convolutional Neural Network (CNN) model from scratch for image classification. I have used a dataset with three classes of digits: 0, 1, and 2. The dataset was split into training and validation sets, and I trained the model on these hand-written digits. Data augmentation techniques were applied to improve the model's performance.

For this task, I took a different approach. Instead of building a model from scratch, I have fine-tuned a pre-trained model on the same custom dataset. Fine-tuning was performed by taking a pre-trained model (VGG16 and ResNetRS50).

### Steps to Perform Transfer Learning
#### Selection of a Pre-trained Model: 
I used VGG16 and ResNetRS50 pre-trained models for image classification. 

#### Prepared the Custom Dataset: 
As in Task 1, I have worked with the same custom dataset containing three classes of digits (0, 1, and 2). I used the same approach of training and validation of dataset.

#### Fine-tuning: 
The key step was to fine-tune the pre-trained model on the custom dataset. This involved modifying the final layers of the pre-trained model according to current classification task.

#### Model Evaluation: 
After training of the fine-tuned model, evaluation has been done by checking its accuracy on the validation dataset.

#### Compare with Task 1: 
Finally, the accuracy of the fine-tuned model was compared with the accuracy achieved in Task 1.

## Task 3: Image Retrieval System using Pre-trained CNN Models
In this task, I have built first image retrieval system. This system allows the users to select a query image from a folder and retrieve the top 4 similar images from a local image database. This task showcases to apply CNN-based feature extraction and practical image retrieval system.

### Python Program: 
This program takes a query image as input and retrieves the top 4 similar images from a local image database. This program utilizes a pre-trained CNN model, such as VGG16 or ResNet, for feature extraction and a similarity metric, like Euclidean distance, for image retrieval.

### Handling Various Image Formats: 
This program is designed to handle images in various formats, including JPG, PNG, and JPEG. This flexibility ensures that users can work with images in the most common formats without limitations.

### Key Components of Your Image Retrieval System
#### Pre-trained CNN Model: 
This program uses th epower of pre-trained CNN models for feature extraction. These models are capable of learning and representing complex visual features in images.

#### Similarity Metric: 
Similarity metric, such as Euclidean distance, to calculate the similarity between the query image and images in the database has been used. This metric allows the system to identify the most visually similar images.

#### Flexibility: 
This program is designed to be flexible in handling different image formats, ensuring that users can work with images in their preferred format.

#### How to Use the Image Retrieval System
To run the image retrieval system and demonstrate its functionality, the following steps were followed:

##### Prerequisites: 
Ensure that the necessary Python libraries are installed, including TensorFlow, OpenCV, and other dependencies used in your program.

##### Prepare Query Images: 
Places the images to uses as queries in the "query_images" folder.

##### Run the Program: 
Execute the Python program, which will prompt to select a query image. Once the query image is selected, the program will retrieve and display the top 4 similar images from the database.

