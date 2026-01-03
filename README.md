# 🏦 End-to-End Bank Loan Default Prediction System

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![SQL](https://img.shields.io/badge/PostgreSQL-ETL%20%26%20Analytics-336791)
![ML](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![Dashboard](https://img.shields.io/badge/Streamlit-Risk%20App-red)
![BI](https://img.shields.io/badge/Tableau-Portfolio%20Analysis-yellow)

## 📖 Executive Summary
In the banking industry, identifying **Non-Performing Assets (NPA)** early is critical for financial stability. This project implements a full-stack data science solution to predict the probability of a client defaulting on a loan.

Using the **Czech Bank Financial Dataset**, this system integrates a Relational Database, an Automated ETL Pipeline, and a Machine Learning Model to provide loan officers with real-time risk assessments and explainable insights.

---

## 📂 About the Data (Czech Bank Dataset)
The data originates from the **PKDD'99 Financial Challenge**, representing real anonymized data from a Czech bank in the late 1990s. It captures a rich relational schema of banking operations.

### The Schema
* **`Account`**: The central entity (Savings/Checking accounts).
* **`Client`**: Demographics (Age, Gender).
* **`Disposition`**: Links clients to accounts (Distinguishing between "Owner" and "Disponent" users).
* **`Transaction`**: The heartbeat of the system. Includes Credit vs. Cash withdrawals, Bank transfers, and **"Sanction Interest"** (Negative balance fines).
* **`Loan`**: The Target Variable.

### The "Target" Variable (Loan Status)
The original dataset uses codes `A`, `B`, `C`, and `D` to describe loan status. In the SQL preprocessing (`bank_database.sql`), I engineered this into a binary target:

| Original Status | Description | Classification |
| :--- | :--- | :--- |
| **A** | Contract finished, no problems | ✅ **Good Loan** |
| **C** | Running contract, OK so far | ✅ **Good Loan** |
| **B** | Contract finished, loan not paid | ❌ **Default (Bad)** |
| **D** | Running contract, client in debt | ❌ **Default (Bad)** |

---

## 🏗️ Technical Architecture

The solution follows a modern data pipeline architecture:

```mermaid
graph LR
    A["Raw CSV Data (Czech Bank)"] -->|"setup.py (ETL)"| B[("PostgreSQL DB")]
    B -->|"SQL Views"| C{"Feature Engineering"}
    C -->|"bank_ml.ipynb"| D["Random Forest Model"]
    D -->|"model.pkl"| E["Streamlit App"]
    B -->|"Real-time Client Data"| E
    B -->|"Data Connector"| F["Tableau Dashboard"]
