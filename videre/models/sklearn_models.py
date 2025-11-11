from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = {
    "lr": LogisticRegression,
    "svm": SVC,
    "mlp": MLPClassifier,
}

def get_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)

def get_model_names():
    return list(models.keys())

def make_lr(**kwargs):
    return LogisticRegression(**kwargs)

def make_svm(**kwargs):
    return SVC(**kwargs)

def make_mlp(**kwargs):
    return MLPClassifier(**kwargs)
