# CNN Image Classification from Scratch – CIFAR-10

## Objective
Build and train a Convolutional Neural Network (CNN) from scratch (without using pre-trained models) to classify images into multiple categories using a publicly available dataset.

This project implements the complete machine learning pipeline:
- Data loading and preprocessing
- CNN model design
- Training with validation
- Evaluation using classification metrics
- Visualization of learning behavior and predictions

---

## Project Description

This project builds an intelligent system that can recognize objects in images using deep learning. The model is trained to look at a picture and automatically identify what it contains, such as an airplane, car, dog, or ship.

A Convolutional Neural Network (CNN) is built from scratch to learn visual patterns from thousands of example images. The system improves by learning from mistakes and gradually becomes better at recognizing different objects.

The project also shows how the model is trained, evaluated, and tested using real data. Performance is measured using accuracy, precision, recall, and a confusion matrix, and example predictions are displayed to show how the model performs on real images.

This project demonstrates how computers can “learn to see” using artificial intelligence and is a complete example of training, testing, and evaluating an image classification model.

---

## Dataset Used
- **Name:** CIFAR-10
- **Total images:** 60,000
  - Training images: 50,000
  - Test images: 10,000
- **Image size:** 32 × 32 × 3 (RGB)
- **Number of classes:** 10  
  (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## Model Architecture
The CNN model was implemented from scratch using TensorFlow/Keras.

### Key components:
- Multiple **Conv2D** layers with ReLU activation  
- **Batch Normalization** for improved training stability  
- **MaxPooling2D** layers for downsampling  
- **Dropout** for regularization  
- A fully connected **Dense** layer  
- Final **Softmax** output layer for classification  

No pre-trained models (ResNet, VGG, MobileNet, etc.) were used.

---

## Training Setup
- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch size:** 64
- **Epochs:** 25  
- **Validation split:** 10%
- **Callbacks used:**
  - EarlyStopping
  - ReduceLROnPlateau

Training and validation accuracy/loss plots are saved in the `results/` folder.

---

## Evaluation & Metrics
The trained model was evaluated on the test dataset.

### Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (raw and normalized)

Example outputs are stored in `results/`:
- `accuracy_curve.png`
- `loss_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `correct_examples.png`
- `incorrect_examples.png`

---

## Results

- **Final Test Accuracy:** 79.96%

### Classification Performance
Precision, Recall, and F1 scores were computed for all classes. Strong performance was observed for:
- Truck (F1 = 0.91)
- Dog (F1 = 0.88)
- Bird (F1 = 0.87)
- Deer (F1 = 0.86)

Classes such as airplane and ship showed lower performance due to visual similarity and low image resolution.

### Observations
- The model generalizes well, as validation accuracy is close to training accuracy.
- No severe overfitting is observed.
- Confusion matrix analysis shows misclassification mainly among visually similar categories.
- The low resolution of CIFAR-10 (32x32) limits fine detail extraction, contributing to confusion among some classes.

---

## Improvements & Tuning Tried
- Added **Dropout** to reduce overfitting.
- Used **Batch Normalization** for stable learning.
- Tuned epoch count and batch size.
- Used learning rate reduction when validation loss plateaued.

---

## Folder Structure

```
CNN_IMAGE_CLASSIFICATION/
├── results/
│   ├── cnn_cifar10_model.keras
│   ├── confusion_matrix_normalized.png
│   ├── confusion_matrix.png
│   ├── correct_examples.png
│   ├── incorrect_example.png
│   ├── traning_validation_accuracy.png
│   └── traning_validation_loss.png
├── evaluate.py
├── model.py
├── README.md
├── requirements.txt
└── train.py
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Evaluate the model
```bash
python evaluate.py
```

---

## Conclusion
This project demonstrates a complete CNN-based image classification pipeline from scratch, including training, evaluation, and result visualization. It reflects a practical understanding of deep learning concepts and performance analysis.

---

By,
Sarvesh Sunil Jagtap.

