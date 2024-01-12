import numpy
import pandas as pd

class z_normVertical():
    def __init__(self,mean,std):
        self.mean=mean.values
        self.std=std.values
    def transform(self,data : numpy.ndarray) -> numpy.ndarray:
        if data is None:
            return None
        data = (data - self.mean) / self.std
        return data
    def transformdf(self,data : pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None
        data = (data - self.mean) / self.std
        return data

class TransformationZnorm():
    def __init__(self,normalize_horizontal=True):
        self.normalize_horizontal=normalize_horizontal

    def transform(self,data : numpy.ndarray) -> numpy.ndarray:
        datao=data
        if data is None:
            return None
        if self.normalize_horizontal:
            data = (data - data.mean()) / (data.std())
        return data
    def transformdf(self,data : pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None
        if self.normalize_horizontal:
            data = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        return data