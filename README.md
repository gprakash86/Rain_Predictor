# Rain Predictor Project
A machine learning project that predicts whether it will rain tomorrow using meterological data from WeatherAUS dataset from Kaggle.
## Features Used
- Minimum Temperature
- Maximum Temperature
- Temperature at 9am
- Temperature at 3pm
- Evaporation
- Sunshine
- Wind Gust speed
- Humidity at 9am
- Humidity at 3pm
- Rainfall
- Cloud at 9am
- Cloud at 3pm
- Pressure at 9am
- Pressure at 3pm
## Data Preprocessing
- Removed missing values.
- Converted rain tomorrow lables from Yes/No to 1/0
- Split data into training and testing sets.
- Used StandardScalar for Logistic Regression.
## Models used
- Decision Tree
- Logical Regression
- Random Forest
## Result
|S.No|Model|Accuracy|
|----|-----|--------|
|1|Decision Tree|79.75%|
|2|Logistic Regression|85.62%|
|**3**|**Random Forest**|**86.23%**|

Random Forest achieved the highest accuracy and is selected as the final model.
## Factors that highly influence the rainfall
- Humidity at 3pm (0.168)
- Sunshine (0.123)
## Technologies used
- Python
- Pandas
- Scikit-learn
- Joblib
## How to run
1. Install the required libraries
   ``` python
       pip install -r requirements.txt
   ```
2. Download WeatherAUS dataset from Kaggle.
3. Run
   ``` python
       python randomforest.py
   ```
