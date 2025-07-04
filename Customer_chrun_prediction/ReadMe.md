# Customer Churn Prediction

This project predicts whether a customer is likely to churn based on various features such as age, subscription type, support interactions, and payment behavior.

## 🔗 Live Demo

Check out the live deployed app in Streamlit :  
👉 Customer Churn Prediction App  https://machine-learning-hskpfddafgdbj8ew4yacfq.streamlit.app/

## 💡 Features

- Predicts churn probability using a trained Random Forest model
- Uses a preprocessing pipeline for transforming user input
- Interactive inputs via Streamlit widgets

## 📁 Files

- `app.py`: Main Streamlit application
- `RandomForestClassifier_best.pkl`: Pickled best-fit model
- `preprocessor.pkl`: Preprocessing pipeline (e.g., `ColumnTransformer`)
- `Chrun_Model.ipynb`: Jupyter notebook for training and evaluation

---

## ⚙️ Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Divya-Chintala/Machine-Learning.git
   cd Machine-Learning/Customer_chrun_prediction

2. Install Dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the app:
     ```bash
     streamlit run app.py

