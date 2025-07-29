# student-performance-prediction
This project aims to predict student academic performance at the beginning of the school year using machine learning. The goal is to classify students into performance tiers—low, average, or high—enabling early interventions and informed decision-making by educators.

📌 Project Objectives
Develop a robust multi-class classification model for student performance forecasting.

Apply and compare various machine learning models.

Identify key factors influencing student performance.

Provide insights to assist educators in proactive academic support.

📊 Dataset
Source: UCI Machine Learning Repository (Cortez & Silva, 2008)

Records: ~650 secondary school students

Features: 30+ demographic, academic, and socioeconomic attributes

Target Variable: performance_tier derived from final grade (G3) into:

Low (≤10)

Average (11–15)

High (16–20)

⚙️ Methodology
1. Preprocessing Pipeline
Missing value imputation (median/mode)

One-hot encoding for categorical features

Standardization of numerical features

Feature selection using RFE and PCA

2. Machine Learning Models
Baseline (majority class)

Decision Tree

Random Forest

Support Vector Machine (SVM)

Neural Network (MLP with ReLU)

3. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation

Evaluation based on accuracy, precision, recall, F1-score

4. Libraries Used
python
Copy
Edit
pandas, numpy, matplotlib, seaborn
scikit-learn (DecisionTreeClassifier, RandomForestClassifier, SVC, MLPClassifier, RFE, GridSearchCV)
TensorFlow/Keras (for MLP implementation)
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the main script:

bash
Copy
Edit
python main.py
(Optional) Load datasets manually:

python
Copy
Edit
math_data = pd.read_csv('student-mat.csv', sep=';')
portuguese_data = pd.read_csv('student-por.csv', sep=';')
📈 Results Summary
Model	Accuracy	Precision	Recall	F1 Score
Baseline	0.468	0.219	0.468	0.298
Decision Tree	0.494	0.495	0.494	0.489
Random Forest	0.494	0.445	0.494	0.467
SVM	0.570	0.531	0.570	0.536
Neural Network	0.544	0.538	0.544	0.532

🔍 Key Insights
Study Time and Absences are among the most influential predictors.

Ensemble models like Random Forest show strong performance with limited data.

Data preprocessing and feature selection significantly impact prediction accuracy.

🔮 Future Work
Develop a real-time prediction dashboard

Integrate additional subjects and schools to reduce bias

Explore time-series models for trend-based forecasting

Design personalized interventions based on predictions

📚 References
Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.

Xiong, H., et al. (2015). Dropout Prediction Using Machine Learning.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.

Chollet, F. (2021). Deep Learning with Python.

