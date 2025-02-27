# 🧠 Stroke Classification Using Deep Learning

This project aims to classify stroke (ischemic or hemorrhagic) using deep learning models applied to MRI/CT scans. Various CNN-based architectures, including EfficientNet variants, AlexNet, and a custom CNN model, have been evaluated to achieve optimal classification performance.

## 📊 Model Performance Comparison

| Model              | Accuracy (%) | Precision | Recall | F1-Score |
|--------------------|-------------|-----------|--------|----------|
| EfficientNet-B2   | 88.5        | 0.87      | 0.89   | 0.88     |
| EfficientNet-B3   | 89.8        | 0.88      | 0.90   | 0.89     |
| EfficientNet-B4   | 90.5        | 0.89      | 0.91   | 0.90     |
| EfficientNet-B5   | 91.0        | 0.90      | 0.92   | 0.91     |
| EfficientNet-B6   | 91.2        | 0.90      | 0.92   | 0.91     |
| AlexNet           | 82.7        | 0.80      | 0.83   | 0.81     |
| Custom CNN        | 79.4        | 0.76      | 0.79   | 0.77     |

### 🔍 Key Observations:
- **EfficientNet-B6 achieved the highest accuracy (91.2%)**, indicating that deeper architectures enhance feature extraction for stroke detection.
- **AlexNet and the custom CNN underperformed**, suggesting that deeper models with optimized parameters generalize better.
- **A trade-off exists between performance and computational efficiency**, as larger EfficientNet models require more computational power.

---

## 🛠 Technologies Used
- **Deep Learning Framework:** TensorFlow / Keras
- **Pretrained Models:** EfficientNet (B2-B6), AlexNet
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib
- **Hardware:** Trained using GPU (NVIDIA CUDA-enabled)

---

## 🔬 Dataset & Preprocessing
- **Dataset:** Publicly available stroke MRI/CT dataset (source to be mentioned if applicable).
- **Preprocessing:**
  - Images resized to **224x224** pixels.
  - Applied **data augmentation** techniques to improve generalization.
  - Normalization for enhanced model convergence.

---

## 🚀 Model Training & Evaluation
- **Training Setup:**
  - Optimizer: **Adam**
  - Loss Function: **Binary Cross-Entropy**
  - Learning Rate Scheduling applied
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Training Time:** Varied across models, with EfficientNet-B6 requiring the longest training duration.

---

## 📌 Future Plans
✔ Further optimization using **hyperparameter tuning**  
✔ Exploring additional **augmentation techniques**  
✔ Testing with **larger datasets for improved generalization**  
✔ Deploying as a **real-time stroke detection tool**  
