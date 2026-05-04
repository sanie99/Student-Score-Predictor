# 🎯 Student Score Predictor

A machine learning project that predicts student exam scores based on lifestyle and study-related factors like study hours, attendance, and more.

This project demonstrates the end-to-end ML workflow — from data preprocessing to deployment using a simple interactive UI built with Streamlit.

---

## 🚀 Project Overview

The goal of this project is to understand how different factors influence academic performance and build a predictive model using Random Forest Regressor.

Users can input their daily habits (study hours, attendance, english score, maths score, etc) and instantly get a predicted score.

---

## 🧠 Features

- 📊 Predict exam scores based on input features
- 🧹 Handles missing data and preprocessing
- ⚖️ Feature scaling for better model performance
- 📈 Model evaluation using MSE & R² score
- 💻 Interactive UI using Streamlit
- 🔄 Reproducible ML pipeline

---

## 🛠️ Tech Stack

- Programming Language: Python
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
- Deployment/UI: Streamlit

---

## 📂 Project Structure

```bash
Student-Score-Predictor/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── Student_Performance.csv
│   ├── processed/
│   │   ├── scaled_student_performance.csv
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── train-test/
│       ├── train_data.csv
│       └── test_data.csv
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_model_comparison.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Workflow

1. Data Collection
2. Data Cleaning (Handling missing values)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Scaling
6. Model Training (Linear Regression)
7. Evaluation (RMSE, R² Score)
8. Deployment using Streamlit

---

## Model Download

HuggingFace Repo with all models used: [link](https://huggingface.co/sanie99/student-score-prediction-rf-model/tree/main)

---

## 📈 Model Performance

| Model Used               | Metric | Value  |
| ------------------------ | ------ | ------ |
| Linear Regressor         | MSE    | ~16.27 |
| Linear Regressor         | R²     | ~0.954 |
| Support Vector Regressor | MSE    | ~16.57 |
| Support Vector Regressor | R²     | ~0.953 |
| Random Forest Regressor  | MSE    | ~8.484 |
| Random Forest Regressor  | R²     | ~0.976 |

---

## 💻 Streamlit App

### Run Locally:

pip install -r requirements.txt  
streamlit run app.py

---

## 🎮 UI Usage

- Enter study hours 📚
- Enter attendance 😴
- Add other inputs
- Click **Predict OverAll Score** → Get score 🎯

---

## 🧾 Example Input

| Feature      | Value |
| ------------ | ----- |
| Study Hours  | 5     |
| Attendance % | 78    |
| Maths Score  | 80    |

Predicted Score: 78.5

---

## 🔍 Key Learnings

- Importance of data preprocessing
- Impact of feature scaling on regression models
- Understanding evaluation metrics
- Building end-to-end ML pipelines
- Deploying ML models with a UI

---

## 📌 Future Improvements

- Add more features (stress level, sleep hours, etc.)
- Add model explainability (SHAP)

---

## 🤝 Contributing

Feel free to fork and improve this project.

---

## 📬 Contact

- GitHub: [link](https://github.com/sanie99)
- LinkedIn: [link](https://www.linkedin.com/in/sanjeevani-sahare-50178737a)

---

## ⭐ If you found this useful

Give this repo a star ⭐
