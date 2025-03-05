import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys

# Custom exception class to capture errors
class CustomException(Exception):
    def __init__(self, message, sys):
        self.message = message
        self.sys = sys

def load_and_check_data(filepath):
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any() > 0:
            print("Missing values detected:")
            print(missing_values)
        else:
            print("No missing values detected")

        # Check data types
        print("\nData types:")
        print(df.dtypes)

        # Check categorical columns
        print("\nCategorical columns:")
        for col in df.select_dtypes(include=['object']).columns:
            print(f"Column '{col}' has categorical values")
        
        return df
    
    except Exception as e:
        raise CustomException(f"Error in loading data: {str(e)}", sys)

def check_train_test_split_and_transformation(df):
    try:
        # Replace 'math_score' with the column you are trying to predict (e.g., 'reading_score', 'writing_score')
        X = df.drop('math_score', axis=1)  # Drop target column
        y = df['math_score']  # Target column

        # Train-test split (80-20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify categorical columns for encoding
        categorical_columns = X.select_dtypes(include=['object']).columns

        # Define a preprocessing pipeline for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical data
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables
        ])

        # Define a full column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_columns)
            ], remainder='passthrough'
        )

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor())  # You can replace this with any model
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae}")

    except Exception as e:
        raise CustomException(f"Error in train-test split or transformation: {str(e)}", sys)

def run_checks():
    try:
        # Load and check the data from the 'artifacts' folder
        df = load_and_check_data('artifacts/data.csv')  # Path to the data file in the artifacts folder

        # Check for train-test split and transformations
        check_train_test_split_and_transformation(df)

    except CustomException as ce:
        print(ce.message)

if __name__ == "__main__":
    run_checks()
