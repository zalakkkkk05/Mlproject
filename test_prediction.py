# test_prediction.py

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Sample input data
custom_data = CustomData(gender='female', race_ethnicity='group A', parental_level_of_education='high school',
                         lunch='standard', test_preparation_course='none', reading_score=72, writing_score=83)

# Get data in the correct format
data_df = custom_data.get_data_as_data_frame()

# Create the prediction pipeline object
predict_pipeline = PredictPipeline()

# Get predictions
predictions = predict_pipeline.predict(data_df)

# Print predictions
print(predictions)
