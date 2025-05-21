# earthquake-rf

This Python script for earthquake magnitude prediction uses a Random Forest Regressor machine learning model. The script performs the following key operations:

1. Imports necessary libraries, including pandas, numpy, scikit-learn components, and visualisation tools like matplotlib and seaborn.

2. Loads earthquake data from a CSV file covering 1995-2023, then preprocesses it by:
   - Dropping 'title' and 'date_time' columns
   - One-hot encoding categorical features
   - Handling missing values

3. Splits the data into training and testing sets with an 80/20 ratio.

4. Applies feature scaling using StandardScaler.

5. Implements a Random Forest Regressor with hyperparameter tuning using GridSearchCV, exploring different combinations of estimators, max depth, and minimum samples for splitting.

6. Evaluates the model using multiple metrics:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² Score

7. Creates several visualisations to analyse model performance:
   - Actual vs. Predicted magnitude plot
   - Residual analysis plots
   - Feature importance visualisation
   - Confusion matrix using binned magnitude values

The code is structured to train a model that can predict earthquake magnitudes based on various features in the dataset.

