import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from TextNormalizer import TextNormalizer

class movie():

    def transformer(self,X):
        #X = pd.Series(X)
        print("Starting.....................................................................")
        X=X.apply(lambda x:TextNormalizer().transform(x))
        print("Data Cleaning...........................................................")
        return X
    def mlflow_run(self,params,register=False,verbose=False):
       with mlflow.start_run() as run:
          print("mlflow running............")
          mlflow.log_params(params)
  
          df = pd.read_excel("train.xlsx")
          #df["Reviews"]=df["Reviews"].apply(lambda x:utils.get_clean(x))
  
          X = df["Reviews"]
          y = df["Sentiment"]
    
          func = FunctionTransformer(self.transformer)
          #tfidf = TfidfVectorizer(max_features=5000)
          #X = tfidf.fit_transform(X)
          
          print("Splitting the data................")
          x_train,x_test, y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
  
          # Train and fit the model
          #lsvc = LinearSVC(**params)
          #lsvc.fit(x_train, y_train)
          #x_test = tfidf.transform(x_test)
          #y_pred = lsvc.predict([x_test])
  
          print("model initializing..................")
          model = Pipeline([
             ('vectorizer',func),
             ('tfidf',TfidfVectorizer()),
             ('trainer',LinearSVC(**params))])

          print("model fitting .....................................")
  
          model.fit(x_train,y_train)
  
          y_pred = model.predict(x_test)
          print(y_pred)
          # Compute metrics
          #mse = mean_squared_error(y_pred, y_test)
          #rsme = np.sqrt(mse)
           
          print("logging.........................................")
          # log params and metrics
          mlflow.log_params(params)
          #mlflow.log_metric("mse", mse)
          #mlflow.log_metric("rmse", rsme)
  
          
          # Specify the `registered_model_name` parameter of the
          # function to register the model with the Model Registry. This automatically
          # creates a new model version for each new run
          mlflow.sklearn.log_model(
             sk_model=model,
             artifact_path="sklearn-model",
             registered_model_name=model_name) if register else mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model")

          run_id = run.info.run_id
  
          return run_id

if __name__ == "__main__":
   # Use sqlite:///mlruns.db as the local store for tracking and registery
   mlflow.set_tracking_uri("mysql+pymysql://nishkarsh:nishkarsh@nishkarsh.cmuzdwd6qin5.ap-south-1.rds.amazonaws.com:3306/nishkarsh")

   # Train, fit and register our model
   params_list = [
      {"C": 0.1}]

   # Iterate over few different tuning parameters
   model_name = "MovieModel"
   for params in params_list:
      model = movie()
      print("Using paramerts={}".format(params))
      runID = model.mlflow_run(params,register=True)
      #print("MLflow run_id={} completed with MSE={} and RMSE={}".format(runID, model.mse, model.rsme))
      print("Mlflow run_id={}".format(runID))

   # Load test data

   #df = pd.read_excel("test.xlsx")
   #df["Reviews"]=df["Reviews"].apply(lambda x:utils.get_clean(x))
   #X = df["Reviews"]
   #X = tfidf.transform(X)
#
   ## Our JSON payload for scoring the model
   ## Use as payload on the REST call to the deployed model
   ## on the local host
   #X.to_json(orient="records")
   print("Done...................")


