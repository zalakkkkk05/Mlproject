from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("D:/mlproject/artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing numerical values
                    ("scaler", StandardScaler())  # Scale numerical data
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing categorical values
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # OneHotEncode categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical data without centering
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Preprocessor with both numerical and categorical transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            print("Preprocessor created successfully!")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            print(f"Reading train data from: {train_path}")
            train_df = pd.read_csv(train_path)
            print(f"Reading test data from: {test_path}")
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Ensure the preprocessing object is created
            if preprocessing_obj is None:
                raise CustomException("Preprocessing object creation failed")

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply preprocessing to input features
            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with the target variables
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Log the transformation completion
            logging.info("Data transformation completed successfully.")

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            print(f"Saving the preprocessor object at {self.data_transformation_config.preprocessor_obj_file_path}")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            # Print the paths for confirmation
            print(f"Train array shape: {train_arr.shape}")
            print(f"Test array shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

# Test if code works by adding a main function and checking output
if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_data_path = 'D:/mlproject/artifacts/train.csv'  # Change this path as needed
    test_data_path = 'D:/mlproject/artifacts/test.csv'  # Change this path as needed

    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
