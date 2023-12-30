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
#params = yaml.safe_load(open("/home/kda/ML_lab3/MLops_lab3/params.yaml"))["train"]
#p_seed = params["seed"]
#p_max_depth = params["max_depth"]

#Define hyperparameters
max_depth =  [10,20,30]
max_features = [10,20,30]
min_samples_leaf = [5,10,15]
params = {}

scp = pd.read_csv("/home/kda/ML_lab3/MLops_lab3/datasets/stage4/train.csv")
scp.info()

X_train = scp.drop(columns=["Price (in USD)"])
y_train = scp["Price (in USD)"]

#Train the model
#clf = DecisionTreeClassifier(**params)
#clf.fit(X,y)

#Evaluate the model
def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test,y_pred)
    return score


def tune_hyperparams(p_max_depth,p_max_features,p_min_samples_leaf):
    params["max_depth"] = int(p_max_depth)
    params["max_features"]= int(p_max_features)
    params["min_samples_leaf"]= int(p_min_samples_leaf)
    with mlflow.start_run():
        mlflow.log_params(params)
        clf = DecisionTreeClassifier(max_depth = params["max_depth"],max_features = params["max_features"],min_samples_leaf=params["min_samples_leaf"])
        mlflow.sklearn.log_model(clf,artifact_path="sklearn-model",
                                 registered_model_name="clf")
        mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/evaluate.py",
                            artifact_path="tuning code")
        clf.fit(X_train,y_train)
        print(clf.score(X_train,y_train))
        mlflow.log_metric("score",clf.score(X_train,y_train))
        mlflow.end_run


for depth in max_depth:
    for feature in max_features:
        for leaf in min_samples_leaf:
            tune_hyperparams(depth,feature,leaf)


    


#with mlflow.start_run():
#    mlflow.sklearn.log_model(clf,
#                             artifact_path="lr",
#                             registered_model_name="lr")
#    mlflow.log_params(params)
#    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/model_learn.py",
#                        artifact_path="model_learn code")
#    mlflow.end_run()

#model = "model.pkl"
#full_model_path = os.path.join(dir_path,model)
#with open(full_model_path,"wb") as fd:
#    pickle.dump(clf,fd)


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

'''mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accuracy_score")
with mlflow.start_run():
    metr= 0
    with open(input_model,"rb") as ff:
        unpickler = pickle.Unpickler(ff)
        tree = unpickler.load()
        scr = tree.score(x_test,y_test)
        metr = scr
    score = clf.score(x_test,y_test)
    mlflow.log_metric("accuracy",score)
    mlflow.log_params(params)
    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/model_learn.py",
                        artifact_path="model_learn code")
    model_info = mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name="eval",
        artifact_path="eval code"
    )
    mlflow.log_artifact(local_path="/home/kda/ML_lab3/MLops_lab3/scripts/evaluate.py",
                        artifact_path="eval code")
    mlflow.end_run()

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(x_test)

with open(output,"w") as f:
    l = {"score":score}
    json.dump(l,f)
print(score)'''

