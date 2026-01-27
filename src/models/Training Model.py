import pandas as pd 
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.model_selection import train_test_split


df = pickle.load(open('data/interim/dataset_level3_feature_ready.pkl' , 'rb'))

x = df.drop(columns=['runs_x'])
y = df['runs_x']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 ,random_state=1)


trf = ColumnTransformer(
    transformers=[
    ('trf' , OneHotEncoder(drop='first' , sparse_output=False) ,  ['batting_team' , 'bowling_team' , 'city'])
]
    ,
    remainder = 'passthrough'
)

pipe = Pipeline(steps = [
    ('step 1' , trf),
    ('step 2' , StandardScaler()),
    ('step 3' , XGBRegressor(n_estimators=1000 , learning_rate=0.2 , max_depth=12 , random_state=1))
])


pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print(r2_score(y_test , y_pred))
print(mean_absolute_error(y_test, y_pred))


pickle.dump(pipe , open('model/pipe.pkl' , 'wb')) 

