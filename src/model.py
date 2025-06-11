import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_model(X, y, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        model = SVC()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)  # use joblib here

def load_model(path):
    return joblib.load(path)  # use joblib here
