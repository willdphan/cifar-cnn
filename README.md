# CIFAR-10 CNN

## Description
This code implements a deep learning model based on a custom Convolutional Neural Network (CNN) architecture for image classification. The model is trained and tested on the CIFAR-10 dataset, which includes images from 10 different classes. The training process involves iterating over mini-batches of images, calculating losses using the CrossEntropyLoss function, and updating the model parameters using the Stochastic Gradient Descent (SGD) optimizer.

During the testing phase, the code loads the test dataset and iterates over the images. Each image is passed through the trained model, which predicts the class of the image. The predicted class is then compared with the true class to calculate the model's accuracy.

## Code

Some important highlights of this code include the use of ReLU activation function, MaxPooling for down-sampling, and two fully connected layers in the end to classify the images into 10 different classes. After defining the model architecture, the model is trained using the SGD optimizer, and the loss function used is CrossEntropyLoss, which is suitable for multi-class classification problems.

This model gives the flexibility of defining the layers and hyper-parameters manually based on intuition and knowledge of the dataset.

[Code](CIFAR_10_CNN.ipynb)

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
