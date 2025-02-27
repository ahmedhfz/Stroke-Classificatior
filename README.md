# 🧠 Stroke Classification Using Deep Learning

This project aims to classify stroke using deep learning models applied to MRI/CT scans. Various CNN-based architectures, including EfficientNet variants, AlexNet, and a custom CNN model, have been evaluated to achieve optimal classification performance.

These experiments and models were developed as part of our preparation and testing for the 2025 Teknofest competition. Additionally, my three teammates are working on different pre-trained models to explore various approaches and improve overall performance.

## 📊 Model Performance Comparison

| Model              | Accuracy (%) | F1-Score | Val Loss (%) |
|--------------------|--------------|----------|-------------|
| EfficientNet-B2   | 94.5         | 0.94     | 12.9        |
| EfficientNet-B3   | 96.7         | 0.96     | 16.8        |
| EfficientNet-B4   | 95.6         | 0.95     | 11.2        |
| EfficientNet-B6   | 97.6         | 0.98     | 9.07        |
| Custom CNN        | 94.8         | ----     | 10.3        |

### 🔍 Key Observations:
- **EfficientNet-B6 achieved the highest accuracy (97.6%)**, indicating that deeper architectures enhance feature extraction for stroke detection.
- **EfficientNet-B3 had slightly higher validation loss (16.8%)**, suggesting overfitting to the training data.
- **A trade-off exists between performance and computational efficiency**, as larger EfficientNet models require more computational power.
- **Custom CNN performed decently but lacked the complexity needed for optimal generalization.**

---

## 🛠 Technologies Used
- **Deep Learning Framework:** TensorFlow / Keras
- **Pretrained Models:** EfficientNet (B2, B3, B4, B6), AlexNet
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib
- **Hardware:** Trained using Google Colab with GPU acceleration

---

## 🔬 Dataset & Preprocessing
- **Dataset:** 2021 Teknofest "Sağlıkta Yapay Zeka" competition dataset.
- **Preprocessing:**
  - Each model was trained using **its optimized input size** rather than a fixed resolution (e.g., EfficientNet-B6 requires a higher resolution than B2).
  - Applied **data augmentation** techniques, including random rotations, flips, and contrast adjustments, to improve generalization.
  - Normalization techniques were applied based on the model’s expected input format.

---

## 🚀 Model Training & Fine-Tuning
- **Transfer Learning & Fine-Tuning:**
  - Used **pretrained EfficientNet models** and applied **transfer learning** by freezing the first few hundred layers.
  - **Fine-tuning** was performed by unfreezing the later layers to allow domain-specific feature extraction.
- **Training Setup:**
  - Optimizer: **Adam**
  - Loss Function: **Binary Cross-Entropy**
  - **Learning Rate Scheduling** applied for better convergence.
  - **Batch size and epoch count optimized** for each model to balance performance and training efficiency.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

## 📌 Future Plans
✔ Further optimization using **hyperparameter tuning**
✔ Exploring additional **augmentation techniques**
✔ Testing with **larger datasets for improved generalization**
✔ Deploying as a **real-time stroke detection tool**
