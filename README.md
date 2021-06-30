# airbnbNY

>## Intro
This is a machine learning model pipeline display, using kaggle dataset: New York City Airbnb Open Data. The task is to predict house pricing. 
In the ipython notebook, it demonstrates process of data cleaning, data exploration analysis, and feature extraction. 

In ml_script.py file, it leverages Linear regression model and Random Forest Regressor model and can set randomizes search cross validation. I use Linear regreiion model as a baseline, so cross validation is only set for Random Forest Regressor model. 

>## The ml_script.py usage

```
>python ml_script.py --model RF --cv 2 --iteration 3
```
This means we train Random Forest Regressor model with cross validation 2 folds and each fold we randomized search 3 times.
Below is parameter setting.
```
>python ml_script.py  -h
usage: ml_script.py [-h] [--model {LR,RF}] [--cv CV] [-i N_ITER]

optional arguments:
  -h, --help            show this help message and exit
  --model {LR,RF}       Linear regression model and Random Forest Regressor model
  --cv CV               random search cross validation folds
  -i N_ITER, --iteration N_ITER
                        random search each fold iteration

```
