
import grand
import numpy as np
from grand import IndividualAnomalyInductive


class grandInductive_core():
    def __init__(self,non_comformity = "knn", k = 20):
        self.non_comformity=non_comformity
        self.k=k
        self.model = IndividualAnomalyInductive(non_conformity = self.non_comformity, k = self.k)
    def fit(self,X_fit):
        self.model.fit(X_fit)

    def predict(self,timestamp,data):
        info=self.model.predict(timestamp, data)
        #return info[2]
        return info[0][2]





