import mlflow
logged_model = './mlruns/0/7872f17549a843bbade74bf1a189b132/artifacts/sklearn-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = ["Veer-Zara is the best romantic movie i have seen . SRK and Preeti Zinta acting was great in it"]
# Predict on a Pandas DataFrame.
import pandas as pd
print(loaded_model.predict(pd.DataFrame(data)))