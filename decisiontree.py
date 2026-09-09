import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#read the file from the provided data(converted to a csv file for convenience)
data = pd.read_csv("Dataset.csv")

#Target variable will be the precipitation probability as I will attempt to create a model that takes features and determines the chance of rain
y = data.precipprob
#Choose the features that I believe influence the chance for rain
x = data[["tempmax", "tempmin", "temp", "co2", "dew", "humidity", "windgust", "windspeed", "winddir", "pressure", "cloudcover", "visibility", "solarradiation", "uvindex"]]

#Create training data and validation data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

#Create the model
model = DecisionTreeRegressor(random_state=1)

#Train the model
model.fit(train_x, train_y)

#Create the predictions
predictions = model.predict(val_x)