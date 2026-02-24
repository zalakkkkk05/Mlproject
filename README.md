# Student Performance Prediction — End-to-End ML Pipeline

**Predicting student exam scores using socioeconomic and parental factors**

---

## Project Overview

This project demonstrates a complete, production-grade machine learning pipeline — from raw data ingestion through preprocessing, model training, evaluation, and Flask deployment. The goal is to predict student math scores based on demographic and socioeconomic features.

The project prioritises **engineering best practices**: modular code, reusable Scikit-learn pipelines, automated testing, and clean separation of concerns — reflecting how real data science teams build and ship models.

---

## Problem Statement

Educational institutions want to identify at-risk students early enough to intervene. By analysing parental education level, lunch type, test preparation status, and other factors, this model surfaces key drivers of academic performance and generates score predictions to guide support programmes.

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student gender |
| `race_ethnicity` | Categorical | Ethnic group (A–E) |
| `parental_level_of_education` | Categorical | Highest education level |
| `lunch` | Categorical | Standard vs free/reduced (socioeconomic proxy) |
| `test_preparation_course` | Categorical | Completed prep course or not |
| `reading_score` | Numerical | Reading exam score |
| `writing_score` | Numerical | Writing exam score |
| **`math_score`** | **Target** | **Variable being predicted** |

---

## Model Comparison

| Model | R² Score |
|---|---|
| Linear Regression | 0.88 |
| Ridge Regression | 0.88 |
| Random Forest | 0.85 |
| XGBoost | 0.87 |
| **CatBoost** | **0.89** |
| AdaBoost | 0.85 |

---

## Tech Stack

- **Language:** Python
- **ML:** Scikit-learn, CatBoost, XGBoost, AdaBoost
- **Pipelines:** Scikit-learn ColumnTransformer + Pipeline (saved as artifacts)
- **Web Framework:** Flask + Jinja2 Templates
- **Testing:** Custom test scripts for load and prediction validation

---

## Project Structure

```
Mlproject/
├── artifacts/             # Trained model and preprocessor (.pkl files)
├── notebook/              # EDA + model training Jupyter notebooks
├── src/
│   ├── components/        # Data ingestion, transformation, model trainer
│   ├── pipeline/          # Prediction pipeline used by Flask
│   └── utils.py
├── templates/             # Flask HTML templates
├── app.py                 # Flask entry point
├── check_data.py          # Data validation
├── check_preprocessors.py # Preprocessing validation
├── test_load.py           # Load testing
├── test_prediction.py     # Prediction tests
└── requirements.txt
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/zalakkkkk05/Mlproject.git
cd Mlproject

# Install dependencies
pip install -r requirements.txt
python setup.py install

# Run the Flask app
python app.py
```

Visit `http://localhost:5000`

```bash
# Run tests
python test_load.py
python test_prediction.py
```

---

## Key Learnings

- Scikit-learn `Pipeline` objects prevent data leakage and make preprocessing reproducible
- CatBoost handles categorical variables natively, removing the need for manual encoding in production
- Separating ingestion, transformation, and training into distinct components makes the code testable and maintainable
- Saving the fitted preprocessor as an artifact alongside the model is critical for consistent inference

---

## Author

**Zalak Patel** — Data Analyst & ML Engineer

[LinkedIn](https://linkedin.com/in/zalak-patel-2989621a1) | [GitHub](https://github.com/zalakkkkk05) | pzalak1234@gmail.com
