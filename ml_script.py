from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set(style="ticks", color_codes=True)
#import statsmodels.api as sm
from scipy import stats
import time
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--model",type=str,choices=["LR","RF"],help="Linear regression model and Random Forest Regressor model", dest="model_name")
parser.add_argument("--cv",type=int, help="random search cross validation folds",dest="cv")
parser.add_argument("-i","--iteration", type=int, help="random search each fold iteration", dest="n_iter")
args = parser.parse_args()
print(args)


# Data loading and examining null value
dataset_raw = pd.read_csv('AB_NYC_2019.csv')


print("dataset_raw.shape: ", dataset_raw.shape)
dataset = dataset_raw.copy()
print("dataset.columns: ", dataset.columns)
print("dataset null: ",dataset.isnull().sum())


# Fill null with tempory value
dataset.fillna({'reviews_per_month':0}, inplace=True)
dataset.fillna({'name':"NoName"}, inplace=True)
dataset.fillna({'host_name':"NoName"}, inplace=True)
dataset.fillna({'last_review':"NotReviewed"}, inplace=True)
print("dataset null ")
print(dataset.isnull().sum())


# Feature observation and extraction 
print('dataset["price] stats:',dataset["price"].describe())
dataset = dataset.drop(columns = ["id","host_name"])
# Try to make more feature 
dataset["name_length"]=dataset['name'].map(str).apply(len)
# Because minimum_nights is severerly skewed, we cut maximun at 30
dataset.loc[(dataset.minimum_nights >30),"minimum_nights"]=30
dataset.drop(["host_id","name",'last_review',"latitude",'longitude'], axis=1, inplace=True)
# We use 300 as threshold price 
dataset=dataset[dataset["price"]<300]



# Categorical encoding
dataset_onehot1 = pd.get_dummies(dataset, columns=['neighbourhood_group',"room_type"], prefix = ['ng',"rt"],drop_first=True)
dataset_onehot1.drop(["neighbourhood"], axis=1, inplace=True)
print("dataset_onehot1.shape: ",dataset_onehot1.shape)


# Train,test dataset split
X1= dataset_onehot1.loc[:, dataset_onehot1.columns != 'price']
Y1 = dataset_onehot1["price"]
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.20, random_state=42)



# Model fitting
def Linear_regression(x_train,y_train,x_test, y_test):
    reg = LinearRegression().fit(x_train, y_train)
    ### R squared value
    r2 = reg.score(x_train, y_train)
    y_pred = reg.predict(x_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return r2, rmse, reg  



# Find most relevant /ilrelevant features 
def Coeff_dataframe(x_train,model, grid):
    Coeff = pd.DataFrame(columns=["Variable","Coefficient"])
    Coeff["Variable"]=x_train.columns
    if grid:
        Coeff["Coefficient"]=model.best_estimator_.feature_importances_
    else:
        Coeff["Coefficient"]=model.coef_
    Coeff.sort_values("Coefficient")
    return Coeff



# Try Random Forest tree with Cross validation 
def Random_Forest_CV(x_train,y_train,x_test, y_test,n_iter, cv):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 20)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    # Create the random param grid
    rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    # Use the random grid to to cross validation
    rf = RandomForestRegressor()
    # Random search of parameters, using K fold cross validation,
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rm_grid, n_iter = n_iter, cv = cv, verbose=1, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    y_pred= rf_random.predict(x_test)
    r2 = rf_random.score(x_train, y_train)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    #print("best_params:",rf_random.best_params_)
    return r2, rmse, rf_random


if __name__ == "__main__":
    model_name = args.model_name
    n_iter = args.n_iter
    cv = args.cv
    if model_name == "LR":
        LR_training_r2, LR_test_rmse, LR_model = Linear_regression(x_train1,y_train1,x_test1, y_test1)
        LR_Coeff = Coeff_dataframe(x_train1, LR_model, grid=False)
        print("LR_training_r2: ", LR_training_r2)
        print("LR_test_rmse: ", LR_test_rmse)
        print("LR_coeff: ",LR_Coeff)
    if model_name == "RF":
        RF_training_r2, RF_test_rmse, RF_model = Random_Forest_CV(x_train1,y_train1,x_test1, y_test1,n_iter, cv)
        RF_Coeff = Coeff_dataframe(x_train1, RF_model, grid=True)
        print("RF_training_r2: ", RF_training_r2)
        print("RF_test_rmse: ", RF_test_rmse)
        print("RF_coeff: ",RF_Coeff)

