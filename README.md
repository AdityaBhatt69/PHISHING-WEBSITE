# Phishing Website Detection Using Supervised Machine Learning

**Author:** Aditya Bhatt (URN: 2481184)  
**Course:** Applied Machine Learning (Semester 5)  
**Institution:** School of Computer Science & Engineering, IILM University, Gurugram  

## Overview
This project implements an end-to-end supervised machine learning pipeline to classify websites as **phishing (`y = 1`)** or **legitimate (`y = 0`)** using the UCI Phishing Websites dataset (11,054 websites, 30 security features), paired with a real-time URL inspection web application (**PhishGuard**).

## Key Highlights
- **Primary Evaluation Metric:** Recall on the phishing class (`y = 1`), motivated by a 10:1 asymmetric misclassification cost (`Cost = 10*FN + 1*FP`).
- **Best Model:** Tuned Random Forest classifier at decision threshold `0.35`, achieving **97.65% Recall**, **95.03% Precision**, and **0.9970 ROC-AUC** on the held-out test set (`n = 2,210`).
- **Leak-Free Pipeline:** `SimpleImputer(strategy='median')` and `StandardScaler()` fitted strictly within 5-fold stratified cross-validation folds.
- **Live Demo (PhishGuard):** Combines a 9-feature URL-only Random Forest model with a character 3–5 gram TF-IDF + domain feature Logistic Regression model and trusted-domain filtering without visiting suspect links.

## Contents
- `Applied_ML_Project_Report_Aditya_Bhatt_2481184.docx` — Complete project report.
- `README.md` — Repository documentation.
