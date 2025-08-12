Advanced Student Score Prediction 📊
This project analyzes the "Student Performance Factors" dataset to predict student exam scores. It demonstrates a complete machine learning workflow, progressing from a simple single-variable regression model to a more complex and accurate model utilizing multiple features.

✨ Key Features
Data Cleaning: Handles missing values and standardizes data formats.

Exploratory Data Analysis (EDA): Visualizes the data distribution and relationships between features using Matplotlib and Seaborn.

Simple Linear Regression: Builds a baseline model to predict exam scores based solely on hours studied.

Polynomial Regression: Explores non-linear relationships as a bonus task.

Advanced Modeling: Implements a RandomForestRegressor that uses multiple numerical and categorical features for significantly improved prediction accuracy.

Feature Importance: Analyzes and visualizes which factors have the most significant impact on student scores.

Model Evaluation: Compares models using key regression metrics like Mean Absolute Error (MAE) and R-squared (R 
2
 ).

Dataset 📚
This project uses the Student Performance Factors dataset. It contains various demographic, parental, and school-related factors that influence student performance.

Source: Kaggle

File: Student_performance_data.csv

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.7+

Git

Installation
Clone the repository:

Bash

git clone https://github.com/YourUsername/YourProjectName.git
cd YourProjectName
Install the required libraries:
A requirements.txt file is recommended for Python projects. You can create one with the following content:

# requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
Install the libraries using pip:

Bash

pip install -r requirements.txt
Add the dataset:
Download Student_performance_data.csv from the Kaggle link above and place it in the root directory of the project.

Usage
To run the complete analysis and model training pipeline, execute the main script from your terminal:

Bash

python Student_Performance.py
The script will print the results and display visualizations for each stage of the analysis.

📈 Model Performance & Results
The project evaluates three different models to demonstrate the impact of feature selection and model complexity. The performance on the test set is summarized below:

Model	R-squared (R 
2
 ) Score	Description
Simple Linear Regression	~0.19	Baseline model using only hours_studied.
Polynomial Regression	~0.20	A slight improvement over the linear model.
Advanced RandomForest	~0.91	A powerful model using multiple relevant features.

Export to Sheets
Key Findings
The final RandomForestRegressor model, which incorporates features like previous_scores, attendance, and parental_involvement, provides a vastly superior prediction.

The feature importance analysis reveals that previous scores and attendance percentage are the most significant predictors of a student's final exam score.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.