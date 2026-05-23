# 🌍 Tourism Experience Analytics
### Classification, Prediction & Recommendation System

A data-driven tourism analytics system that predicts attraction ratings, 
classifies tourist visit modes, and recommends personalized attractions 
using Machine Learning — deployed as an interactive Streamlit web application.

---

## 🚀 Live App
👉 [Tourism Analytics App](https://tourism-analytics-lspgujljb57fg6tfgdkxtg.streamlit.app/)

---

## 📌 Project Overview

| Property | Details |
|---|---|
| Domain | Tourism & Travel Analytics |
| Language | Python 3.12 |
| Development | Google Colab, VS Code |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 🎯 Objectives

- Predict tourist attraction ratings using regression models
- Predict tourist visit mode using classification models
- Recommend personalized attractions using a hybrid recommendation system
- Deploy all models in an interactive Streamlit web application

---

## 📊 Dataset

The dataset consists of 9 interlinked relational tables:

| Table | Rows | Description |
|---|---|---|
| Transaction | 52,930 | Core visit records with ratings |
| User | 33,530 | Tourist demographic information |
| City | 9,143 | City reference data |
| Country | 165 | Country reference data |
| Region | 22 | Regional groupings |
| Continent | 6 | Continental groupings |
| Updated Item | 1,698 | Attraction details |
| Type | 17 | Attraction type categories |
| Mode | 6 | Visit mode labels |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web Application | Streamlit |
| Development | Google Colab, VS Code |
| Version Control | GitHub |

---

## 🔄 Project Workflow

Data Cleaning → Merging & Feature Engineering → EDA →
Model Building → Evaluation → Streamlit Deployment

---

## 🤖 Machine Learning Models

### 1. Regression — Predict Rating
Predicts the rating (1–5) a tourist will give to an attraction.

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 0.2903 | 0.4972 | **0.7375 🏆** |
| XGBoost | 0.2719 | 0.5020 | 0.7324 |
| Random Forest | 0.2759 | 0.5414 | 0.6888 |

### 2. Classification — Predict Visit Mode
Predicts how a tourist travels — Couples, Family, Friends, Solo or Business.

| Model | Accuracy | F1 Score |
|---|---|---|
| XGBoost | 0.4623 | **0.4565 🏆** |
| LightGBM | 0.4590 | 0.4483 |
| Random Forest | 0.4451 | 0.4492 |

> Class imbalance handled using SMOTE — all 5 classes balanced to 17,296 samples

### 3. Recommendation System
Hybrid system combining:
- **Collaborative Filtering** — recommends based on similar user behavior
- **Content Based Filtering** — recommends based on attraction similarity

---

## 📁 Project Structure

tourism-analytics/
├── data/
│   ├── raw/                    ← original Excel files
│   └── cleaned/                ← cleaned CSV files + master dataset
├── notebooks/
│   └── Tourism_Analytics.ipynb ← complete project notebook
├── models/
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── label_encoder.pkl
│   ├── user_item_matrix.pkl
│   ├── attraction_similarity.pkl
│   └── rec_scaler.pkl
├── app/
│   └── app.py                  ← Streamlit application
├── docs/
│   └── Tourism_Analytics_Report.docx
├── requirements.txt
└── README.md

---

## 🖥️ Streamlit App Features

| Page | Description |
|---|---|
| 🏠 Home | Key metrics — total visits, users, attractions, avg rating |
| 📊 EDA Dashboard | Interactive charts — Ratings, Visits, Geography |
| ⭐ Predict Rating | Predict attraction rating based on user profile |
| 🧳 Predict Visit Mode | Predict travel mode based on visit details |
| 🎯 Get Recommendations | Get personalized attraction suggestions |

---

## 📈 Key Insights

- 🌟 Average tourist rating is **4.16/5** — tourists are generally satisfied
- 💑 **Couples dominate** tourism at 40% of all visits
- 🏖️ **Beaches and Nature** attractions are most popular
- 🌏 **Asia and Europe** contribute highest tourist activity
- 📅 **Season has minimal impact** on tourist satisfaction
- 🔧 **Feature engineering** was the biggest driver of model performance

---

## 👩‍💻 Author

**Sabitha J**
Domain: Data Science with Gen AI

---
