from thresholding import thresholding
from utils.structure import Datapoint,Eventpoint,PredictionPoint
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt
class PairDetection():
    def __init__(self,thresholdtype,thresholdfactor,actualProfileSize=1,constThresholdfilter=float('-inf'),alarmsThreshold=0,normz=False):
        self.score_buffer=[]
        self.thresholdfactor = thresholdfactor
        self.thresholdtype = thresholdtype
        self.reference=None
        self.ProfileSize=actualProfileSize
        self.constThresholdfilter=constThresholdfilter
        self.initilized=False
        self.pbcore=None
        self.alarmsThreshold=alarmsThreshold
        self.normz=normz
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

        description=""
        counter=0
        producealarm=0
        for th,score in zip(pairthresholds,pair_anomaly_scores):
            if score>th and score>self.constThresholdfilter:
                producealarm+=1
                #description+=f"{counter},"
            if len(thdeatails)>0:
                description+=f"{thdeatails[counter][0]},{thdeatails[counter][1]},{score},"
            else:
                description += f"{th},{score},"
            counter+=1
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
            self.pbcore = pairDetectionCore(profile, thresholdtype=self.thresholdtype,dataforThreshold=dataForNormal,thresholdfactor=self.thresholdfactor,normz=self.normz)
        else:
            profile=self.reference
            self.pbcore = pairDetectionCore(profile, thresholdtype=self.thresholdtype,thresholdfactor=self.thresholdfactor,normz=self.normz)

    def reset(self):
        self.score_buffer = []
        self.reference = None
        self.initilized = False
        self.pbcore = None



class pairDetectionCore():
    def __init__(self,profile, thresholdtype="selftunne",dataforThreshold=None,thresholdfactor=5,normz=False):
        self.profile=profile
        self.normz=normz
        self.dims=len(profile[0][0])
        self.profileTrees=[]
        for i in range(self.dims):
            # extrac first dimesnion vectors
            tempdatadim=[]
            for point in self.profile:
                temppointi = point.transpose()[i]
                if self.normz==True:

                    if temppointi.std()==0:
                        sample = temppointi - temppointi.mean()
                    else:
                        sample=(temppointi - temppointi.mean()) / temppointi.std()

                    tempdatadim.append(sample)
                else:
                    tempdatadim.append(temppointi)
            self.profileTrees.append(KDTree(tempdatadim,leaf_size=5))

        self.thresholdfactor=thresholdfactor
        self.thresholdtype=thresholdtype
        self.dataforThreshold=dataforThreshold
        self.thdetails=[]
        if thresholdtype=="inner":
            self.threshold=self.calculateThresholdinner()
        elif thresholdtype=="selftunne" and dataforThreshold is not None:
            self.threshold,self.thdetails=self.calculateThresholdSelfTune(dataforThreshold)
    def calculateThresholdinner(self):
        if len(self.profile)<=1:
            assert False, " PairDetection needs more than one data point for inner threshold Calculation"
        finalthresholds=[]
        for i in range(len(self.profile)):
            temp=[]
            counter=0
            for point in self.profile:
                if counter!=i:
                    temp.append(point)
                counter+=1
            dists=self.calculateDistsMany(temp, self.profile[i])
            if len(finalthresholds) == 0:
                finalthresholds = dists
            else:
                finalthresholds = [max(dd, fd) for dd, fd in zip(dists, finalthresholds)]
        finalthresholds=[self.thresholdfactor*fth for fth in finalthresholds]
        return finalthresholds

    def calculateThresholdSelfTune(self,dataforThreshold):
        anomalyscoresinNormal = [self.predict(point) for point in dataforThreshold]

        anomalyscoresinNormal=np.array(anomalyscoresinNormal)
        finalthresholds=[]
        thdetails = []
        for i in range(self.dims):
            pairthreshold,thmean,thstd = thresholding.selfTuning(factor=self.thresholdfactor,anomalyscoresInNormal=anomalyscoresinNormal[:,i],returnmean=True)
            finalthresholds.append(pairthreshold)
            thdetails.append((thmean,thstd))
        return finalthresholds,thdetails
    def fit(self, profile):
        self.profile=profile
    # profile is [[[3,4,5],[3,4,5],[3,4,5]],
    #       [[3,4,5],[3,4,5],[3,4,5]],
    #       .
    #       .
    #       .
    #       ]
    #data is
    #     [[3,4,5],[3,4,5],[3,4,5]]
    def predict(self,data):
        if len(self.profile)==1:
            return self.predictFromOne(data)
        else:
            #return self.predictFromMultiple(data)
            tempdata=data.transpose()
            dists=[]
            for i in range(len(tempdata)):
                if self.normz == True:
                    if tempdata[i].std()==0:
                        sample = tempdata[i] - tempdata[i].mean()
                    else:
                        sample = (tempdata[i] - tempdata[i].mean()) / tempdata[i].std()
                    disti,inds=self.profileTrees[i].query([sample],k=1)
                else:
                    disti, inds = self.profileTrees[i].query([tempdata[i]], k=1)
                disti=disti[0]
                dists.append(disti[0])
            return dists
    def predictFromOne(self,data):
        return self.calculateDists(self.profile[0],data)


    def distanceTimeseries(self,x,y):
        if len(x)>3:
            # plt.subplot(121)
            # plt.plot(x)
            # plt.plot(y)
            #x=(x-x.mean())/x.std()
            #y=(y-y.mean())/y.std()
            # plt.subplot(122)
            # plt.plot(x)
            # plt.plot(y)
            # plt.show()
            d=np.linalg.norm(x - y)
        else:
            d=np.linalg.norm(x - y)
        return d
    # profile is [[3,4,5],[3,4,5],[3,4,5]]
    # data is
    #     [[3,4,5],[3,4,5],[3,4,5]]
    def calculateDists(self,a,b):
        a=a.transpose()
        b=b.transpose()
        dists = [ self.distanceTimeseries(x,y) for x, y in zip(a, b)]
        return dists
    # a is 2d aaray and b is 1D
    def calculateDistsMany(self,a,b):
        finaldists = []
        for pf in a:
            dists = self.calculateDists(pf, b)
            if len(finaldists) == 0:
                finaldists = dists
            else:
                finaldists = [min(dd, fd) for dd, fd in zip(dists, finaldists)]
        return finaldists
    def predictFromMultiple(self,data):
        return self.calculateDistsMany(self.profile,data)