import pandas as pd
import sys
import os
import io

os.makedirs('data/stage3', exist_ok=True)

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage2/Sport_car_price.csv")
Car_Model = pd.get_dummies(scp["Car Model"], drop_first=True)
scp = scp.drop("Car Model", axis = 1)
scp = pd.concat([scp, Car_Model], axis = 1)

scp.to_csv('/home/kda/ML_lab3/MLops_lab3/datasets/stage3/Sport_car_price.csv')
