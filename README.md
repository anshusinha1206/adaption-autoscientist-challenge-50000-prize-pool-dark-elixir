# adaption-autoscientist-challenge-50000-prize-pool-dark-elixir
Hackathon team repository for Dark Elixir - [hackindia-team:adaption-autoscientist-challenge-50000-prize-pool:dark-elixir]
# 🫀 Early Heart Disease Risk Stratification Model

## 📌 Project Overview
Cardiovascular diseases are a leading cause of global mortality. This project aims to automate the risk stratification of patients using a machine learning pipeline. By analyzing clinical parameters, vital statistics, and lifestyle factors, this model serves as a diagnostic aid to help medical professionals identify and prioritize high-risk patients instantly. 

The primary metric of success for this medical model is **Recall**, ensuring that False Negatives (predicting a sick patient as healthy) are minimized.

## 📊 Dataset
* **Source:** Highly robust dataset containing 70,000 patient records (`heart_disease_risk_dataset_earlymed.csv`).
* **Features:** Includes continuous variables (Age) and binary categorical variables (Chest Pain, High BP, Obesity, Smoking, etc.).
* **Target:** `Heart_Risk` (0.0 = Low Risk, 1.0 = High Risk).

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest)
* **Serialization:** Pickle

## 🚀 Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
* Analyzed the distribution of the target variable to ensure class balance.
* Generated a Feature Correlation Matrix to identify which physiological factors mathematically drove the highest risk probabilities.

### 2. Data Preprocessing
* **Stratified Splitting:** Divided the data into a 56,000-patient Training Set (80%) and a 14,000-patient Testing Set (20%), preserving the high/low-risk ratio.
* **Feature Scaling:** Applied `StandardScaler` to the continuous `Age` column to normalize its variance against the binary (0/1) symptom columns.

### 3. Model Training & Evaluation
Two models were trained and evaluated on the hidden 14,000-patient test set. The **Random Forest Classifier** was selected as the final model due to its exceptional ability to capture non-linear relationships between interacting symptoms.

**Random Forest Performance:**
* **Accuracy:** > 99%
* **Precision:** > 99% (Minimal False Positives)
* **Recall:** > 99% (Only 64 missed cases out of 14,000 patients, drastically minimizing dangerous False Negatives).

### 4. Model Serialization
The trained Random Forest model and the StandardScaler were bundled and exported as a `heart_disease_model.pkl` file for backend deployment and real-time patient inference.

## 💻 How to Run (Locally)
1. Clone this repository.
2. Install the required dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter Notebook to view the EDA and training pipeline.
4. Use the `heart_disease_model.pkl` file in your Python backend for inference.
