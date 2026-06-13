## Skin Cancer Detection Using Deep Learning

### Overview

SafeSkin AI is a deep learning-based web application developed for the early detection of skin cancer from dermoscopic images. The system analyzes skin lesion images and predicts whether the lesion is **Benign** or **Malignant**, helping in early screening and diagnosis.

The project uses the **HAM10000 dataset**, which originally contains seven different types of skin lesions. For this project, the dataset was converted into a **binary classification problem** by grouping the lesions into two categories:

* Benign
* Malignant

To improve prediction accuracy and reliability, an ensemble approach combining **DenseNet201** and **InceptionV3** was implemented.

---

## Features

* Skin cancer detection from dermoscopic images
* Binary classification (Benign / Malignant)
* Deep Learning Ensemble using DenseNet201 and InceptionV3
* Transfer Learning based implementation
* Grad-CAM visualization for Explainable AI
* Real-time prediction through FastAPI
* User-friendly web interface using Bootstrap
* Automatic PDF report generation
* Report history management

---

## Tech Stack

* Python
* FastAPI
* TensorFlow / Keras
* DenseNet201
* InceptionV3
* OpenCV
* NumPy
* HTML
* CSS
* Bootstrap
* JavaScript

---

## Model Details

* **Dataset:** HAM10000
* **Original Classes:** 7 Skin Lesion Categories
* **Classification:** Binary (Benign vs Malignant)
* **Training Technique:** Transfer Learning
* **Models Used:**

  * DenseNet201
  * InceptionV3
* **Ensemble Method:** Soft Voting

Final prediction is obtained by averaging the outputs of both models and applying an optimized classification threshold.

---

## Explainable AI

The project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight the regions of the image that influence the model's prediction. This improves transparency and helps users understand the AI's decision-making process.

---

## Performance

* Accuracy: **91.3%**
* AUC Score: **0.93**

---

## Workflow

1. Upload a dermoscopic skin image.
2. Preprocess the image.
3. Generate predictions using DenseNet201.
4. Generate predictions using InceptionV3.
5. Combine predictions using Soft Voting Ensemble.
6. Classify the lesion as Benign or Malignant.
7. Generate a Grad-CAM visualization.
8. Display results and generate a PDF report.

---

## Future Improvements

* Multi-class skin disease classification
* Mobile application support
* Cloud deployment
* Doctor dashboard integration
* Real-time telemedicine support

---

## Contributors

* Kajal Kumari
* Annu Kumari 
* Sushant Sagar
* Abhishek Choudhary

