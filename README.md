🏡 Gurugram House Predictions
This project focuses on predicting housing prices in Gurugram using machine learning. It includes the full pipeline from data preprocessing to model training, evaluation, and automated inference using saved models. The goal is to build a reliable price prediction system based on location, income, and housing features.

📌 Project Objective
Predict median_house_value in Gurugram.

Build and save a complete ML pipeline.

Automate predictions on unseen data.

Provide a reusable and scalable solution for house price prediction.

🧰 Tools & Technologies Used
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Joblib

Techniques:

Stratified Sampling

Data Cleaning and Feature Engineering

Model Training and Cross-Validation

Model & Pipeline Saving

Inference on Test Data

🗂 Project Structure
housing.csv: Main dataset

input.csv: 20% test data for prediction

output.csv: Predictions output file

model.pkl: Trained Random Forest model

pipeline.pkl: Full data preprocessing pipeline

housing_model.py: Main script

README.md: Project documentation

🔄 Workflow Summary
Data Loading: Load the housing dataset.

Stratified Sampling: Based on income categories to ensure balanced train-test split.

Preprocessing:

Numerical features: Imputation and scaling

Categorical features: One-hot encoding

Pipeline Creation: Combines all preprocessing steps.

Model Training: Train multiple models and select the best (Random Forest).

Evaluation: RMSE with 10-fold cross-validation.

Saving Artifacts: Save trained model and pipeline.

Inference: Predict values on test data and save results.

📊 Model Comparison Summary
Model	Performance Summary
Linear Regression	High error
Decision Tree	Overfitting
Random Forest ✅	Best accuracy and generalization

📁 Key Files
housing.csv: Dataset used for training/testing

input.csv: Test set (automatically created on first run)

output.csv: Final predictions

model.pkl: Trained Random Forest model

pipeline.pkl: Preprocessing pipeline

housing_model.py: All logic (training + inference)

README.md: Project documentation

🧠 Learning Outcomes
Build real-world ML pipelines using Scikit-learn

Handle missing data, scaling, and categorical encoding

Train, evaluate, and compare multiple models

Save/load models for production use

Automate predictions using saved pipeline

🔮 Future Enhancements
Add hyperparameter tuning with GridSearchCV

Visualize model performance and feature importance

Deploy with a web interface using Flask/Streamlit

Improve exception handling and input validation

Extend to other city datasets

📄 License & Credits
Dataset: Based on California housing dataset; adapted for project demonstration

License: Open-source 

Author: Piyush Tripathi
