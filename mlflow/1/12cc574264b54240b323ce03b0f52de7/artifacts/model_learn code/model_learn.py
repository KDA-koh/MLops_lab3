import sys
import os
import yaml
import pickle
import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

dir_path = "/home/kda/ML_lab3/MLops_lab3/datasets/models"
os.makedirs(dir_path,exist_ok=True)
params = yaml.safe_load(open("/home/kda/ML_lab3/MLops_lab3/params.yaml"))["train"]
p_seed = params["seed"]
p_max_depth = params["max_depth"]

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage4/train.csv")
scp.info()

X = scp.drop(columns=["Price (in USD)"])
y = scp["Price (in USD)"]


clf = DecisionTreeClassifier(max_depth=p_max_depth, random_state=p_seed)

with mlflow.start_run():
    mlflow.sklearn.log_model(clf,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/model_learn.py",
                        artifact_path="model_learn code")
    mlflow.end_run()

clf.fit(X,y)


model = "model.pkl"
full_model_path = os.path.join(dir_path,model)
with open(full_model_path,"wb") as fd:
    pickle.dump(clf,fd)
