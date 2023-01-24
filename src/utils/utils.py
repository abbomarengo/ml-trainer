import pickle
import os


def load_history(model_dir):
    path = os.path.join(model_dir, 'history', "history.pkl")
    history = pickle.load(open(path, "rb"))
    return history