import os
import sys
from dataclasses import dataclass
import pandas as pd  # Import pandas to handle dataframes
import numpy as np  # Import numpy as np


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("D:/mlproject/artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Debugging: Type Checking
            if not isinstance(train_array, np.ndarray):
                raise CustomException("train_array should be a NumPy array")
            if not isinstance(test_array, np.ndarray):
                raise CustomException("test_array should be a NumPy array")

            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {'learning_rate': [.1, .01, .05, .001], 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Linear Regression": {},
                "XGBRegressor": {'learning_rate': [.1, .01, .05, .001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "CatBoosting Regressor": {'depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100]},
                "AdaBoost Regressor": {'learning_rate': [.1, .01, 0.5, .001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            logging.info(f"Model Evaluation Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error(f"Error in Model Trainer: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Executing ModelTrainer...")

    train_file_path = "D:/mlproject/artifacts/train.csv"
    test_file_path = "D:/mlproject/artifacts/test.csv"

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    train_array = train_df.to_numpy()
    test_array = test_df.to_numpy()

    logging.info(f"Loaded train data shape: {train_array.shape}")
    logging.info(f"Loaded test data shape: {test_array.shape}")

    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_array, test_array)
    logging.info(f"Best Model R2 Score: {r2_square}")

