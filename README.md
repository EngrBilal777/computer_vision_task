# Computer Vision Tasks

## Task - 1: Hand-Written Digits Classification with TensorFlow CNN
This project implemented a Convolutional Neural Network (CNN) model from scratch using TensorFlow for image classification. The dataset used for this task consisted of hand-written digits with three classes: 0, 1, and 2. The goal was to train a model to classify these digits accurately.

### Dataset
The dataset was organized in the following format:

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
In the code, the dataset was split into training and validation sets. 80% of the data was used for training, and the remaining 20% was used for validation. TensorFlow's built-in functions were utilized to perform this split.

### Data Augmentation
Data augmentation was performed to enhance the dataset's diversity and robustness.

### Model Architecture
The model architecture used for this task was a customizable CNN. Achieving the best possible accuracy was the primary objective while making CNN architecture.

### Training
The model was trained on the provided hand-written digits dataset. Training was conducted for 10 epochs by default. The task was evaluated based on the validation accuracy.

### Results
After training the model for 10 epochs, the achieved accuracy on the validation set is reported. 


## Task 2: Fine-tuning a Pre-trained Model for Image Classification
In this task,fine tuning of the pretrained model have focused on transfer learning, making this task an opportunity to solidify  understanding of transfer learning and to fine-tune a pre-trained model for custom dataset. The main objective was to observe the accuracy of the fine-tuned model and compare it with the accuracy achieved in Task 1.

### Task Overview
In Task 1, I have implemented a basic Convolutional Neural Network (CNN) model from scratch for image classification. I have used a dataset with three classes of digits: 0, 1, and 2. The dataset was split into training and validation sets, and I trained the model on these hand-written digits. Data augmentation techniques were applied to improve the model's performance.

For this task, I took a different approach. Instead of building a model from scratch, I have fine-tuned a pre-trained model on the same custom dataset. Fine-tuning was performed by taking a pre-trained model (VGG16 and ResNetRS50).

### Steps to Perform Transfer Learning
#### Selection of a Pre-trained Model: 
I used VGG16 and ResNetRS50 pre-trained models for image classification. 

#### Custom Dataset: 
As in Task 1, I have worked with the same custom dataset containing three classes of digits (0, 1, and 2). I used the same approach of training and validation of dataset.

#### Fine-tuning: 
The key step was to fine-tune the pre-trained model on the custom dataset. This involved modifying the final layers of the pre-trained model according to current classification task.

#### Model Evaluation: 
After training of the fine-tuned model, evaluation has been done by checking its accuracy on the validation dataset.

#### Compare with Task 1: 
Finally, the accuracy of the fine-tuned model was compared with the accuracy achieved in Task 1.

## Task 3: Image Retrieval System using Pre-trained CNN Models
In this task, I have built an image retrieval system. This system allows the user to select a query image from a folder and retrieve the top 4 similar images from a local image database. This task showcases to apply CNN-based feature extraction and practical image retrieval system.

### Python Program: 
This program took a query image as input and retrieved the top 4 similar images from a local image database. This program utilized a pre-trained CNN model, such as VGG16 or ResNet, for feature extraction and a similarity metric, like Euclidean distance, for image retrieval.

### Handling Various Image Formats: 
This program was designed to handle images in various formats, including JPG, PNG, and JPEG. This flexibility ensures that users can work with images in the most common formats without limitations.

### Key Components of Your Image Retrieval System
#### Pre-trained CNN Model: 
This program used the power of pre-trained CNN models for feature extraction. These models were capable of learning and representing complex visual features in images.

#### Similarity Metric: 
Similarity metric, such as Euclidean distance, to calculate the similarity between the query image and images in the database was used. This metric allowed the system to identify the most visually similar images.

#### Flexibility:  
This program was designed to be flexible in handling different image formats, ensuring that users can work with images in their preferred format.

#### How to Use the Image Retrieval System
To run the image retrieval system and demonstrate its functionality, the following steps were followed:

##### Prerequisites: 
Ensured that the necessary Python libraries were installed, including TensorFlow, OpenCV, and other dependencies used in the program.

##### Prepare Query Images: 
Placed the images to be used as queries in the "query_images" folder.

##### Run the Program: 
Executed the Python program, which prompted to select a query image. Once the query image was selected, the program retrieved and displayed the top 4 similar images from the database.
