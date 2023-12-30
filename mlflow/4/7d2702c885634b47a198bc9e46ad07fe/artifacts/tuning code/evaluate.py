import sys
import os
import yaml
import pickle
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

dir_path = "/home/kda/ML_lab3/MLops_lab3/datasets/models"
os.makedirs(dir_path,exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tree_model_train")

#Define hyperparameters
params = {}

max_depth =  [10,20,30]
max_features = [10,20,30]
min_samples_leaf = [5,10,15]


#Evaluate the model

test_file= os.path.join("datasets","stage4","test.csv")
input_model = "/home/kda/ML_lab3/MLops_lab3/datasets/model.pkl"
output = os.path.join("metrics","eval.json")
os.makedirs(os.path.join("metrics"),exist_ok=True)

df_scp = pd.read_csv(test_file)
print(df_scp)
x_test = df_scp.drop(labels=["Price (in USD)"],axis=1)
y_test = df_scp["Price (in USD)"]

params = yaml.safe_load(open("params.yaml"))["train"]
p_max_depth = params["max_depth"]
p_max_features = params["max_features"]
p_min_samples_leaf = params["min_samples_leaf"]

metr= 0
with open(input_model,"rb") as ff:
    unpickler = pickle.Unpickler(ff)
    tree = unpickler.load()
    scr = tree.score(x_test,y_test)
    metr = scr

with open(output,"w") as f:
    l = {"score":metr}
    json.dump(l,f)
print(metr)
