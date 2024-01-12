import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from thresholding import thresholding

from operator import itemgetter

from utils.structure import Datapoint, PredictionPoint


class PairXgboost():
    def __init__(self,thresholdtype,thresholdfactor,actualProfileSize=30,constThresholdfilter=float('-inf'),alarmsThreshold=0):
        self.score_buffer=[]
        self.thresholdfactor = thresholdfactor
        self.thresholdtype = thresholdtype
        self.reference=None
        self.ProfileSize=actualProfileSize
        self.constThresholdfilter=constThresholdfilter
        self.initilized=False
        self.pbcore=None
        self.alarmsThreshold=alarmsThreshold
    # Data point should contain (reference,actual data,source) and the function return PredictionPoint
    def get_data(self,point : Datapoint):
        anomaly_score=0
        if self.initilized:
            pair_anomaly_scores=self.pbcore.predict(point.current)
        else:
            self.initilize(point)
            pair_anomaly_scores = self.pbcore.predict(point.current)

        alarm = False
        pairthresholds = self.pbcore.threshold
        thdeatails = self.pbcore.thdetails

        description = ""
        counter = 0
        producealarm = 0
        for th, score in zip(pairthresholds, pair_anomaly_scores):
            if score > th and score > self.constThresholdfilter:
                producealarm += 1
                # description+=f"{counter},"
            if len(thdeatails) > 0:
                description += f"{thdeatails[counter][0]},{thdeatails[counter][1]},{score},"
            else:
                description += f"{th},{score},"
            counter += 1
        alarm = producealarm>self.alarmsThreshold
        if alarm:
            anomaly_score=1
        else:
            anomaly_score=0
        prediction = PredictionPoint(anomaly_score, 0.5, alarm,self.thresholdtype,
                                     point.timestamp,point.source,notes=description,ensemble_details=(pairthresholds,pair_anomaly_scores))
        return prediction

    def initilize(self,point : Datapoint):
        self.reference=point.reference
        self.createpbCore()
        self.initilized=True


    def createpbCore(self):
        if self.thresholdtype=="selftunne":
            profile=self.reference[:self.ProfileSize]
            dataForNormal=self.reference[self.ProfileSize:]
            self.pbcore = XgboostFeatureCore(profile, thresholdtype=self.thresholdtype,dataforThreshold=dataForNormal,thresholdfactor=self.thresholdfactor)
        else:
            profile=self.reference
            self.pbcore = XgboostFeatureCore(profile, thresholdtype=self.thresholdtype,thresholdfactor=self.thresholdfactor)

    def reset(self):
        self.score_buffer = []
        self.reference = None
        self.initilized = False
        self.pbcore = None



class XgboostFeatureCore():

    def __init__(self, profile, thresholdtype="selftunne", dataforThreshold=None, thresholdfactor=5):
        self.profile = profile
        self.thresholdfactor = thresholdfactor
        self.thresholdtype = thresholdtype
        self.dataforThreshold = dataforThreshold
        self.models=[]
        self.fit(profile)
        self.thdetails=[]
        if thresholdtype == "const":
            self.threshold = thresholdfactor
        elif thresholdtype == "selftunne" and dataforThreshold is not None:
            self.threshold,self.thdetails= self.calculateThresholdSelfTune(dataforThreshold)

    def calculateThresholdSelfTune(self, dataforThreshold):
        anomalyscoresinNormal = [self.predict(point) for point in dataforThreshold]

        anomalyscoresinNormal = np.array(anomalyscoresinNormal)
        finalthresholds = []
        thdetails = []
        for i in range(len(anomalyscoresinNormal[0])):
            pairthreshold, thmean, thstd = thresholding.selfTuning(factor=self.thresholdfactor, anomalyscoresInNormal=anomalyscoresinNormal[:, i],returnmean=True)
            finalthresholds.append(pairthreshold)
            thdetails.append((thmean, thstd))
        return finalthresholds,thdetails

    def remove_col(self,arr, ith):
        itg = itemgetter(*filter((ith).__ne__, range(len(arr[0]))))
        return list(map(list, map(itg, arr)))
    def fit(self, profile):
        self.profile = profile
        dataforTraining=[]
        self.models=[]
        for seq in self.profile:
            #[[3,4,5],[3,4,5],[3,4,5]]
            dataforTraining.append(seq[-1])
        df = pd.DataFrame(dataforTraining,columns=[i for i in range(len(dataforTraining[0]))])
        for feature in range(len(dataforTraining[0])):
            dftemp=df.copy()
            label=dftemp[feature]
            dftemp=dftemp.drop([feature],axis=1)
            modeltemp=xgb.XGBRegressor()
            modeltemp.fit(dftemp.values, label)
            self.models.append(modeltemp)

    # profile is [[[3,4,5],[3,4,5],[3,4,5]],
    #       [[3,4,5],[3,4,5],[3,4,5]],
    #       .
    #       .
    #       .
    #       ]
    # data is
    #     [[3,4,5],[3,4,5],[3,4,5]]
    def predict(self, data):
        dataforTraining = data[-1]
        distances=[]
        for feature in range(len(dataforTraining)):
            tempdata=[dataforTraining[i] for i in range(len(dataforTraining)) if i!=feature]
            label=dataforTraining[feature]
            pred=self.models[feature].predict([tempdata])
            distances.append(abs(label-pred[0]))
        return distances
