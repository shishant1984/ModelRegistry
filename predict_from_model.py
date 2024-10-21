import mlflow.pyfunc
import numpy as np

data = np.array([1,22,24,36,7,97,11.2,323,657]).reshape(1,-1)

model_name = "diabetes-rf"
model_version = 2

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}") #Tellinhg which model to use for prediction

print(model.predict(data)) # predict 