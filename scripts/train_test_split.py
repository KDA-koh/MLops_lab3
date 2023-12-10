import yaml
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs('data/stage4', exist_ok=True)
params = yaml.safe_load(open("/home/kda/ML_lab3/MLops_lab3/params.yaml"))["split"]

p_split_ratio = params["split_ratio"]

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage3/Sport_car_price.csv")

X = scp.drop(columns=["Price (in USD)", "Torque (lb-ft)"])
y = scp["Price (in USD)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p_split_ratio, random_state=1)

pd.concat([y_train,X_train],axis=1).to_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage4/train.csv",index=None)
pd.concat([y_test,X_test],axis=1).to_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage4/test.csv",index=None)
