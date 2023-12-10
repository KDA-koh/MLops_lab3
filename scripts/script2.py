import pandas as pd
import sys
import os
import io

os.makedirs('data/stage2', exist_ok=True)

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage1/Sport_car_price.csv")
Car_Make = pd.get_dummies(scp["Car Make"], drop_first=True)
scp = scp.drop("Car Make", axis = 1)
scp = pd.concat([scp, Car_Make], axis = 1)

scp.to_csv('/home/kda/ML_lab3/MLops_lab3/datasets/stage2/Sport_car_price.csv')
