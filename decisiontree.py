import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
#read the file from the provided data(converted to a csv file for convenience)

#Function that takes a maxnode count and returns the mae for that maxnode
def get_mae(maxnodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=maxnodes, random_state=1)
    model.fit(train_x, train_y)
    return(mean_absolute_error(val_y, model.predict(val_x)))

#Function that takes training data to determine the best tree depth
def max_leaf_node_optimization(train_x, val_x, train_y, val_y):
    maxnodes = 2
    best = 100
    bestnodes = maxnodes
    upperbound = 100
    #Test max tree nodes, increasing upperbound if one is found that is better
    while maxnodes < upperbound:
        mae = get_mae(maxnodes, train_x, val_x, train_y, val_y)
        if mae < best:
            best = mae
            bestnodes = maxnodes
            upperbound = upperbound + 5
        maxnodes = maxnodes + 1
    return bestnodes
#Function that takes training data to determine the best features to choose from
def optimize_feature_selection(x, y):
    best = 100
    new_x = []
    for feature in x:
        new_x.append(feature)
        train_x, val_x, train_y, val_y = train_test_split(x[new_x], y, random_state=1)
        model = DecisionTreeRegressor(random_state=1)
        model.fit(train_x, train_y)
        predictions = model.predict(val_x)
        mae = mean_absolute_error(val_y, predictions)
        #If mae doesn't improve, remove the newly added feature
        if mae > best:
            new_x.pop()
        else:
            best = mae
            
    return new_x

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

#Mean Absolute Error, metric 1
mae = mean_absolute_error(val_y, predictions)
print("Unoptimized Mean Absolute Error: ", mae)

#r2, metric 2
r2 = r2_score(val_y, predictions)
print("Unoptimized R2 score: ", r2)



optimized_x = data[optimize_feature_selection(x, y)]
optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y = train_test_split(optimized_x, y, random_state=1)
optimized_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_node_optimization(optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y), random_state=1)
optimized_model.fit(optimized_train_x, optimized_train_y)
optimized_predictions = optimized_model.predict(optimized_val_x)
print(f"Optimized MAE: {mean_absolute_error(optimized_val_y, optimized_predictions)}")
print(f"Optimized R2 score: {r2_score(optimized_val_y, optimized_predictions)}")

