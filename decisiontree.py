import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
#read the file from the provided data(converted to a csv file for convenience)

#Function that takes a parameter and returns the mae for that parameter
def get_mae(train_x, val_x, train_y, val_y, maxnodes=None, maxdepth=None, minsamplesleaf=1):
    model = DecisionTreeRegressor(max_leaf_nodes=maxnodes, max_depth=maxdepth, min_samples_leaf=minsamplesleaf, random_state=1)
    model.fit(train_x, train_y)
    return(mean_absolute_error(val_y, model.predict(val_x)))

#Function that takes training data to determine the best max leaf nodes for the tree
def max_leaf_node_optimization(train_x, val_x, train_y, val_y):
    maxnodes = 2
    best = 100
    bestnodes = maxnodes
    upperbound = 100
    #Test max leaf nodes, increasing upperbound if one is found that is better
    while maxnodes < upperbound:
        mae = get_mae(train_x, val_x, train_y, val_y, maxnodes=maxnodes)
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

#Function that takes training data to determine the best tree depth
def regularize_tree_depth(train_x, val_x, train_y, val_y, maxnodes):
    best = 100
    depth = 2
    bestdepth = depth
    upperbound = 100
    while depth < upperbound:
        mae = get_mae(train_x, val_x, train_y, val_y, maxdepth=depth, maxnodes=maxnodes)
        if mae < best:
            best = mae
            bestdepth = depth
            upperbound = upperbound + 5
        depth = depth + 1
    return bestdepth

#Function that takes training data to determine the best min
def regularize_min_samples(train_x, val_x, train_y, val_y, maxnodes, maxdepth):
    best = 100
    minsamples = 2
    bestmin = minsamples
    upperbound = 100
    while minsamples < upperbound:
        mae = get_mae(train_x, val_x, train_y, val_y, minsamplesleaf=minsamples, maxdepth=maxdepth, maxnodes=maxnodes)
        if mae < best:
            best = mae
            bestmin = minsamples
            upperbound = upperbound + 5
        minsamples = minsamples + 1
    return bestmin

#Function for evaluating models
def evaluate(val_y, predictions, name):
    print()
    print(f"{name} MAE: {mean_absolute_error(val_y, predictions)}")
    print(f"{name} R2 score: {r2_score(val_y, predictions)}")

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

#
evaluate(val_y, predictions, "Unoptimized Decision Tree")

#Use the above optimization and regularization methods to find good parameters
optimized_x = data[optimize_feature_selection(x, y)]
optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y = train_test_split(optimized_x, y, random_state=1)
optimized_max_leaf_nodes = max_leaf_node_optimization(optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y)
regularized_max_depth=regularize_tree_depth(optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y, optimized_max_leaf_nodes)
regularized_min_samples_leaf=regularize_min_samples(optimized_train_x, optimized_val_x, optimized_train_y, optimized_val_y, optimized_max_leaf_nodes, regularized_max_depth)

#Create a DecisionTree with the optimized and regularized parameters
optimized_tree = DecisionTreeRegressor(max_leaf_nodes=optimized_max_leaf_nodes, 
                                        max_depth=regularized_max_depth,
                                        min_samples_leaf=regularized_min_samples_leaf, 
                                        random_state=1)
optimized_tree.fit(optimized_train_x, optimized_train_y)
optimized_tree_predictions = optimized_tree.predict(optimized_val_x)
evaluate(optimized_val_y, optimized_tree_predictions, "Optimized Decision Tree")

#Create a random forest model as ensemble 1
random_forest = RandomForestRegressor(random_state=1)
random_forest.fit(optimized_train_x, optimized_train_y)
random_forest_predictions = random_forest.predict(optimized_val_x)
evaluate(optimized_val_y, random_forest_predictions, "Random Forest")

#Create a gradient boosting model as ensemble 2
gradient_boost = GradientBoostingRegressor(random_state=1)
gradient_boost.fit(optimized_train_x, optimized_train_y)
gradient_boost_predictions = gradient_boost.predict(optimized_val_x)
evaluate(optimized_val_y, gradient_boost_predictions, "Gradient Boosting")