# 🎓 Early Burnout Risk Prediction for Students using Machine Learning

## 📌 Overview

Student burnout is becoming a serious concern due to academic pressure, lifestyle imbalance, and mental health challenges.
This project aims to **predict student burnout levels (Low / Medium / High)** using Machine Learning techniques, enabling **early intervention and prevention**.

---

## 🎯 Objective

* Predict burnout levels of students
* Use academic and lifestyle features
* Assist in early detection of mental health risks
* Compare multiple ML models to find the best performer

---

## 📊 Dataset & Features

* **Dataset Type:** Student Mental Health Dataset
* **Target Variable:** Burnout Level (Low, Medium, High)

### 🔹 Features Used:

* Daily Study Hours
* Daily Sleep Hours
* Screen Time
* Physical Activity Hours
* Anxiety Score
* Depression Score
* Academic Pressure Score
* Stress Level

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Handling categorical variables
   * Feature scaling (StandardScaler)
   * Label encoding

2. **Exploratory Data Analysis (EDA)**

   * Data distribution analysis
   * Class balance check

3. **Feature Engineering**

   * Created `burnout_score` using weighted features
   * Converted into categories using `qcut`

4. **Model Training**

   * Logistic Regression
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Decision Tree
   * Random Forest
   * AdaBoost
   * Gradient Boosting

5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * ROC-AUC
   * Confusion Matrix
   * Cross-validation

---

## 🤖 Models Used

| Model               | Description                         |
| ------------------- | ----------------------------------- |
| Logistic Regression | Linear classification model         |
| SVM                 | Finds optimal separating hyperplane |
| KNN                 | Distance-based classification       |
| Decision Tree       | Rule-based model                    |
| Random Forest       | Ensemble of decision trees          |
| AdaBoost            | Boosting-based model                |
| Gradient Boosting   | Sequential learning model           |

---

## 📈 Results & Performance

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Random Forest       | 0.99     | 0.99     | 1.00    |
| Logistic Regression | 0.98     | 0.98     | 1.00    |
| Gradient Boosting   | 0.98     | 0.98     | 1.00    |
| SVM                 | 0.98     | 0.98     | 1.00    |
| KNN                 | 0.93     | 0.93     | 0.99    |
| Decision Tree       | 0.82     | 0.82     | 0.94    |
| AdaBoost            | 0.84     | 0.84     | 0.89    |

---

## 📊 ROC Curve Analysis

* Most models achieved **AUC ≈ 1.00**, indicating strong class separation
* Ensemble models (Random Forest, Gradient Boosting) performed best
* KNN performed slightly lower due to sensitivity to data distribution

---

## 🔍 Key Insights

* Dataset is well-balanced → reliable predictions
* Lifestyle factors strongly influence burnout
* Stress and sleep are major contributors
* Ensemble models outperform single models

---

## 🧠 Conclusion

Machine Learning models can effectively predict student burnout levels.
Early detection can help institutions take preventive measures and improve student well-being.

---

## 🚀 Future Work

* Implement Deep Learning models
* Use real-world and larger datasets
* Deploy as a web/mobile application
* Integrate with mental health support systems

---

## 📌 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 👩‍💻 Contributors

* Ashmika Khandelwal
* Vaishnavi Shukla
* Lavanya Agarwal

---

## 📎 How to Run

```bash
git clone https://github.com/ashmika139/Early-Burnout-Risk-for-students.git
cd Early-Burnout-Risk-for-students
pip install -r requirements.txt
python your_script.py
```

---

## 📬 Contact

For queries or collaboration, feel free to reach out!
