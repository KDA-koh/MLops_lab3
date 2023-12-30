import sys
import os
import yaml
import pickle
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


test_file= os.path.join("datasets","stage4","test.csv")
input_model = "/home/kda/ML_lab3/MLops_lab3/datasets/models/model.pkl"
output = os.path.join("metrics","eval.json")
os.makedirs(os.path.join("metrics"),exist_ok=True)

#params = yaml.safe_load(open("params.yaml"))["train"]
#p_max_depth = params["max_depth"]
#p_max_features = params["max_features"]
#p_min_samples_leaf = params["min_samples_leaf"]

df_scp = pd.read_csv(test_file)
print(df_scp)
x_test = df_scp.drop(labels=["Price (in USD)"],axis=1)
y_test = df_scp["Price (in USD)"]

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accuracy_score")
with mlflow.start_run():
    metr= 0
    with open(input_model,"rb") as ff:
        unpickler = pickle.Unpickler(ff)
        tree = unpickler.load()
        scr = tree.score(x_test,y_test)
        metr = scr
    mlflow.log_metric("accuracy",metr)
    model_info = mlflow.sklearn.log_model(
        sk_model=tree,
        registered_model_name="eval",
        artifact_path="eval code"
    )
    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/evaluate.py",
                        artifact_path="eval code")
    mlflow.end_run()

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(x_test)

with open(output,"w") as f:
    l = {"score":metr}
    json.dump(l,f)
print(metr)
