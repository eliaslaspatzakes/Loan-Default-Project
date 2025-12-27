# 🏦 End-to-End Bank Loan Default Prediction System

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![SQL](https://img.shields.io/badge/PostgreSQL-Advanced-336791)
![ML](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![Dashboard](https://img.shields.io/badge/Streamlit-Live%20App-red)
![BI](https://img.shields.io/badge/Tableau-Business%20Intelligence-yellow)

## 📖 Executive Summary
In the banking industry, minimizing **Non-Performing Assets (NPA)** is critical for financial stability. This project implements a full-stack data science solution to predict the probability of a client defaulting on a loan.

By integrating a **Relational Database (PostgreSQL)**, an **Automated ETL Pipeline**, and a **Machine Learning Model (Random Forest)**, this system provides loan officers with real-time risk assessments and explainable insights (e.g., *"Client rejected due to high sanction count"*).

---

## 🏗️ Technical Architecture

The solution follows a modern data pipeline architecture:

```mermaid
graph LR
    A[Raw CSV Data] -->|setup.py (ETL)| B(PostgreSQL DB)
    B -->|SQL Views| C{Data Cleaning & Feature Eng.}
    C -->|bank_ml.ipynb| D[Machine Learning Model]
    D -->|model.pkl| E[Streamlit App]
    B -->|Live Queries| E
    B -->|Connector| F[Tableau Dashboard]
