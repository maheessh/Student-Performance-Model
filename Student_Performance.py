# =============================================================================
# 0. SETUP: IMPORT TOOLS & LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("All libraries imported successfully. ✅")


# =============================================================================
# 1. LOAD AND CLEAN THE DATASET
# =============================================================================
try:
    # Task: Use the recommended "Student Performance Factors" dataset
    df = pd.read_csv('StudentPerformanceFactors.csv')
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("\nError: 'Student_performance_data.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory.")
    exit()

# --- Task: Perform data cleaning ---
# Standardize column names (e.g., 'Hours Studied' -> 'hours_studied')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Handle missing values by filling with the most frequent value (mode)
for col in ['teacher_quality', 'parental_education_level', 'distance_from_home']:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

print("Data cleaning complete: Standardized column names and handled missing values.")


# =============================================================================
# 2. SIMPLE LINEAR REGRESSION (SCORE VS. STUDY HOURS)
# =============================================================================
print("\n--- Task 1: Building a model based on Study Hours ---")

# --- Task: Perform basic visualization to understand the dataset ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='hours_studied', y='exam_score', data=df, alpha=0.6)
plt.title('Exam Score vs. Hours Studied', fontsize=15)
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.grid(True)
plt.show()

# --- Task: Split the dataset into training and testing sets ---
X_simple = df[['hours_studied']] # Feature: only study hours
y = df['exam_score']             # Target: the exam score

X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train_simple)} training samples and {len(X_test_simple)} test samples.")

# --- Task: Train a linear regression model ---
linear_model = LinearRegression()
linear_model.fit(X_train_simple, y_train)
print("Linear regression model trained successfully.")

# --- Task: Visualize predictions and evaluate model performance ---
y_pred_linear = linear_model.predict(X_test_simple)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test, color='blue', label='Actual Scores', alpha=0.6)
plt.plot(X_test_simple, y_pred_linear, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Linear Regression: Actual vs. Predicted Scores', fontsize=15)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("\n--- Linear Regression Performance ---")
print(f"Mean Absolute Error (MAE): {mae_linear:.2f}")
print(f"R-squared (R²): {r2_linear:.4f}")


# =============================================================================
# 3. BONUS 1: POLYNOMIAL REGRESSION
# =============================================================================
print("\n\n--- Bonus 1: Trying Polynomial Regression ---")

# Create a pipeline to combine polynomial feature creation and regression
poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg_model", LinearRegression())
])

poly_pipeline.fit(X_train_simple, y_train)
y_pred_poly = poly_pipeline.predict(X_test_simple)

# Evaluation
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print("\n--- Polynomial Regression Performance ---")
print(f"Mean Absolute Error (MAE): {mae_poly:.2f}")
print(f"R-squared (R²): {r2_poly:.4f}")

print(f"\nComparison: Polynomial regression R² ({r2_poly:.4f}) vs. Linear regression R² ({r2_linear:.4f})")
if r2_poly > r2_linear:
    print("Result: Polynomial regression performs slightly better for this feature.")
else:
    print("Result: Simple linear regression is sufficient for this feature.")


# =============================================================================
# 4. BONUS 2: EXPERIMENTING WITH MORE FEATURES
# =============================================================================
print("\n\n--- Bonus 2: Building an Advanced Model with Multiple Features ---")

# Feature selection: Using a combination of numerical and categorical features
numerical_features = ['hours_studied', 'attendance', 'sleep_hours', 'previous_scores']
categorical_features = ['parental_involvement', 'extracurricular_activities', 'gender']

X_advanced = df[numerical_features + categorical_features]

# Create a preprocessing pipeline to handle different data types
# Numerical features will be scaled; categorical features will be one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Create the full model pipeline with preprocessing and a powerful regressor
# RandomForest is a great choice for handling complex interactions
advanced_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data using the new, multi-feature X
X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_advanced, y, test_size=0.2, random_state=42)

# Train the advanced model
advanced_model_pipeline.fit(X_train_adv, y_train_adv)
print("Advanced RandomForest model trained successfully.")

# Evaluate the advanced model
y_pred_advanced = advanced_model_pipeline.predict(X_test_adv)
mae_advanced = mean_absolute_error(y_test_adv, y_pred_advanced)
r2_advanced = r2_score(y_test_adv, y_pred_advanced)
print("\n--- Advanced Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae_advanced:.2f}")
print(f"R-squared (R²): {r2_advanced:.4f}")


# =============================================================================
# 5. FINAL CONCLUSION
# =============================================================================
print("\n\n--- Overall Model Comparison ---")
print(f"Simple Linear Model (Hours Studied only): R² = {r2_linear:.4f}")
print(f"Polynomial Model (Hours Studied only):   R² = {r2_poly:.4f}")
print(f"Advanced Model (Multiple Features):     R² = {r2_advanced:.4f}")
print("\nConclusion: While study hours have a positive correlation with exam scores,")
print("a model using multiple features (like previous scores, attendance, etc.)")
print("provides a significantly more accurate prediction. ✨")