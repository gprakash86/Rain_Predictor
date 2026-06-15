import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/weatherAUS.csv")
df = df.dropna()

X = df[['MinTemp', 'MaxTemp', 'Humidity9am',
        'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'Cloud9am', 'Cloud3pm', 'Rainfall']]

y = df['RainTomorrow']
y = y.map({'No':0, 'Yes':1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)
print(cm)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

feature_importances = model.feature_importances_

for feature, importance in zip(X.columns, feature_importances):
    print(feature, ":", importance)