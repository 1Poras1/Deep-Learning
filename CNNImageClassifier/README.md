# CNN Image Classifier - Cat & Dog

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify images into different categories. The application is built using TensorFlow, Keras, and Streamlit, providing an interactive interface for model training and image prediction.

## Project Overview

This project demonstrates how Convolutional Neural Networks can be used for image classification.

Users can train a CNN model using either the CIFAR-10 dataset or a custom Cats vs Dogs dataset and upload an image to predict its category.

## Features

- **CNN Architecture:** Uses convolutional and pooling layers to extract image features.
- **Multiple Datasets:** Supports CIFAR-10 and custom Cats vs Dogs datasets.
- **Model Training:** Train the CNN model through an interactive interface.
- **Image Prediction:** Upload an image and predict whether it belongs to the Cat or Dog category.
- **Prediction Confidence:** Displays prediction probabilities for the available classes.
- **Training Visualization:** Displays training and validation accuracy and loss graphs.
- **Interactive Interface:** Built with Streamlit for easy interaction.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Streamlit
- Pillow (PIL)

## Project Structure

```text
CNNImageClassifier/
│
├── app.py
├── cnn_model.py
├── requirements.txt
└── README.md
```

## CNN Architecture

The model uses the following layers:

1. Convolutional Layer (32 filters)
2. Max Pooling Layer
3. Convolutional Layer (64 filters)
4. Max Pooling Layer
5. Convolutional Layer (128 filters)
6. Max Pooling Layer
7. Flatten Layer
8. Dense Layer (128 neurons)
9. Dropout Layer (30%)
10. Output Layer (Softmax activation)

**Optimizer:** Adam

**Loss Function:** Categorical Crossentropy

**Evaluation Metric:** Accuracy

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/1Poras1/Deep-Learning.git
```

### 2. Navigate to the Project Folder

```bash
cd Deep-Learning/CNNImageClassifier
```

### 3. Create a Virtual Environment

```bash
python -m venv myenv
```

### 4. Activate the Virtual Environment

Windows:

```bash
myenv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## How to Use

1. Open the Streamlit application.
2. Select the dataset you want to use.
3. Train the CNN model.
4. Wait for model training to complete.
5. Navigate to the prediction section.
6. Upload an image.
7. View the predicted class and prediction confidence.

## Datasets

This project supports:

- **CIFAR-10:** A dataset containing 10 different image categories.
- **Cats vs Dogs:** A custom image dataset containing cat and dog images.

The Cats vs Dogs dataset can be provided as a ZIP file through the application.

## Learning Outcomes

Through this project, I learned:

- Fundamentals of Deep Learning.
- CNN architecture and its working.
- Image preprocessing and normalization.
- Model training and validation.
- Image classification using TensorFlow and Keras.
- Model evaluation using accuracy and loss.
- Building interactive machine learning applications using Streamlit.

## Future Improvements

- Improve prediction accuracy with data augmentation.
- Experiment with deeper CNN architectures.
- Implement transfer learning using pretrained models.
- Save trained models for reuse.
- Deploy the application online.

## Author

**Poras**

GitHub: https://github.com/1Poras1

---

*This project was developed as part of my Deep Learning practical work to understand and implement image classification using Convolutional Neural Networks.*