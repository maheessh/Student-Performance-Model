# 📊 Advanced Student Score Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project analyzes the "Student Performance Factors" dataset to predict student exam scores. It demonstrates a complete machine learning workflow, progressing from a simple single-variable regression model to a more complex and accurate model utilizing multiple features.

---

## ✨ Key Features

- **Data Cleaning:** Handles missing values and standardizes data formats.
- **Exploratory Data Analysis (EDA):** Visualizes data distribution and relationships between features using Matplotlib and Seaborn.
- **Simple Linear Regression:** Builds a baseline model to predict exam scores based solely on hours studied.
- **Polynomial Regression:** Explores non-linear relationships as a bonus task.
- **Advanced Modeling:** Implements a `RandomForestRegressor` that uses multiple numerical and categorical features for significantly improved prediction accuracy.
- **Feature Importance:** Analyzes and visualizes which factors have the most significant impact on student scores.
- **Model Evaluation:** Compares models using key regression metrics like Mean Absolute Error (MAE) and R-squared ($R^2$).

---

## 📚 Dataset

This project uses the **Student Performance Factors** dataset, which contains various demographic, parental, and school-related factors that influence student performance.

- **Source:** [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **File:** `Student_performance_data.csv`

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.7+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourProjectName.git](https://github.com/YourUsername/YourProjectName.git)
    cd YourProjectName
    ```

2.  **Install the required libraries:**
    A `requirements.txt` file is recommended. You can create one with the following content:
    ```
    # requirements.txt
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    ```
    Install the libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add the dataset:**
    Download `Student_performance_data.csv` from the Kaggle link and place it in the root directory of the project.

### Usage

To run the complete analysis and model training pipeline, execute the main script from your terminal:
```bash
python Student_Performance.py
