# MindBridge AI

**A Machine Learning Tool for Mental Health Analysis in Sinhala and Singlish**

## About the Project
MindBridge AI analyzes social media posts in Sinhala and Singlish to detect signs of mental distress. It uses the XLM-RoBERTa model to classify text into four categories: Normal, Mild, Moderate, and Severe. It also uses Explainable AI (LIME) to show exactly which words influenced the model's decision.

## Features
* **Severity Classification:** Detects 4 levels of mental distress.
* **Explainable AI (XAI):** Shows feature importance using LIME.
* **Language Support:** Processes code-mixed Sinhala and Singlish.
* **Expert Feedback:** Collects true labels from professionals.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/Hasithjay98/mindbridge-ai.git](https://github.com/Hasithjay98/mindbridge-ai.git)
   cd mindbridge-ai

 Install dependencies:
  pip install -r requirements.txt 

  Run the application:
  streamlit run app.py