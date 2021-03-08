import pickle
import utils

class IMDb(object):
    def __init__(self):
        print("Initializing class .......")
        self.loaded_model = pickle.load(open("model.pkl",'rb'))
        self.tfidf_model = pickle.load(open("tfidf.pkl",'rb'))
        print("Loading model..........")

    def predict(self,X,feature_name):
        clean_text = utils.get_clean(X)
        print("Data Cleaned .............")
        tfidf_features = self.tfidf_model.transform([clean_text])
        predictions = self.loaded_model.predict(tfidf_features)
        predict_prob =self.loaded_model._predict_proba_lr(tfidf_features)
        print("Predicting result .......................")
        return predictions , predict_prob
