from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df=pd.read_csv("House Price Prediction Dataset.csv")
for c in df.select_dtypes('object'): # Convert String → Numbers
    df[c] = LabelEncoder().fit_transform(df[c])
print("After Encoding:",df.head())
x=df[["Area","Bedrooms","Bathrooms","Floors","YearBuilt","Location","Condition","Garage"]]
y=df["Price"]

model=RandomForestRegressor(n_estimators=100,max_depth=3,random_state=42)
model.fit(x,y)
prediction=model.predict([[1500,3,2,2,2015,1,0,1]])
print(prediction[0])