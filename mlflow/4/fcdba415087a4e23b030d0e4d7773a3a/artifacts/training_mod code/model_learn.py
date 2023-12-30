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

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage4/train.csv")
scp.info()

X_train = scp.drop(columns=["Price (in USD)"])
y_train = scp["Price (in USD)"]

#Train the model
#tree = DecisionTreeClassifier(**params)
#tree.fit(X,y)


def tune_hyperparams(p_max_depth,p_max_features,p_min_samples_leaf):
    params["max_depth"] = int(p_max_depth)
    params["max_features"]= int(p_max_features)
    params["min_samples_leaf"]= int(p_min_samples_leaf)
    print(params)
    with mlflow.start_run():
        mlflow.log_params(params)
        tree = DecisionTreeClassifier(max_depth = params["max_depth"],max_features = params["max_features"],min_samples_leaf=params["min_samples_leaf"])
        mlflow.sklearn.log_model(tree,artifact_path="sklearn-model",
                                 registered_model_name="tree")
        mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/evaluate.py",
                            artifact_path="tuning code")
        tree.fit(X_train,y_train)
        print(tree.score(X_train,y_train))
        mlflow.log_metric("score",tree.score(X_train,y_train))
        mlflow.end_run


for depth in max_depth:
    for feature in max_features:
        for leaf in min_samples_leaf:
            tune_hyperparams(depth,feature,leaf)

with mlflow.start_run():
    mlflow.set_experiment_tag("version","best")
    experements = mlflow.search_runs(experiment_names=["tree_model_train"], order_by=["metrics.score DESC"])
    best_experement = experements.iloc[0]
    print("TESTEST BEST EXPEREMENT:      " + best_experement["run_id"])
    mlflow.log_params(params)
    best_model= mlflow.sklearn.load_model("runs:/" + best_experement["run_id"] + "/sklearn-model")
    best_model.fit(X_train,y_train)
    mlflow.sklearn.log_model(best_model,
                            artifact_path="sklearn-model",
                            registered_model_name="best_tree") 
    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/model_learn.py",
                        artifact_path="training_mod code")
    
    mlflow.log_metric("score", best_model.score(X_train, y_train))
    model = "model.pkl"
    full_model_path = os.path.join(dir_path,model)
    with open(full_model_path, "wb") as f:
        pickle.dump(best_model, f)                        
    mlflow.end_run()
  
