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
import infer_signature

dir_path = "/home/kda/ML_lab3/MLops_lab3/datasets/models"
os.makedirs(dir_path,exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tree_model_test")

#Evaluate the model

test_file= os.path.join("datasets","stage4","test.csv")
input_model = "/home/kda/ML_lab3/MLops_lab3/datasets/models/model.pkl"
output = os.path.join("metrics","eval.json")
os.makedirs(os.path.join("metrics"),exist_ok=True)

#DataFrame for Testing 
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

with mlflow.start_run():
    mlflow.log_metric("score", metr)
    with open(input_model, "rb") as ff: 
        unpickler = pickle.Unpickler(ff)
        tree = unpickler.load()
        model_info = mlflow.sklearn.log_model(tree,
                            artifact_path="sklearn-model",
                            registered_model_name="tree",
                            input_example=x_test)        
        model = mlflow.sklearn.load_model(model_info.model_uri)
        pred = model.predict(x_test)
        signature = infer_signature(x_test, pred)
        mlflow.sklearn.log_model(tree,
                            artifact_path="sklearn-model",
                            registered_model_name="tree",
                            signature = signature,
                            input_example=x_test)  

        mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/evaluate.py",
                            artifact_path="testing_model")
        
    mlflow.end_run()