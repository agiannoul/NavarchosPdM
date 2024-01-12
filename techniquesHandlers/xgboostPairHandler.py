import numpy as np
from numpy import ndarray
from utils.structure import PredictionPoint,Datapoint,Eventpoint
from Techniques import pairdetection,xgboostPerFeature
import pandas as pd

def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

@auto_str
class XgboostPairHandler():
    # inner or selftunne
    def __init__(self,source,thresholdtype="const",thresholdfactor=2,ProfileSize=30,ActualProfileSize=1,resetcodes=[],currentReference=None,constThresholdfilter=float('-inf'),sequencesize=1,alarmsThreshold=0):
        self.profilebuffer=[]
        self.profilebuffertimestamps=[]
        self.sequencesize=sequencesize
        self.alarmsThreshold=alarmsThreshold
        self.pointbuffer=[]
        self.currentReference=currentReference

        self.constThresholdfilter=constThresholdfilter

        self.resetcodes = resetcodes
        self.thresholdtype = thresholdtype
        self.thresholdfactor = thresholdfactor
        self.source=source
        self.ProfileSize=ProfileSize
        if thresholdtype=="const":
            self.ActualProfileSize = ProfileSize
        elif ActualProfileSize<1:
            self.ActualProfileSize=int(ActualProfileSize*self.ProfileSize)
        else:
            self.ActualProfileSize=ActualProfileSize

        self.model = xgboostPerFeature.PairXgboost(thresholdtype, thresholdfactor,self.ActualProfileSize,constThresholdfilter=self.constThresholdfilter,alarmsThreshold=self.alarmsThreshold)


        if self.currentReference is None:
            self.calculated_profile = False
        else:
            temp_datapoint = Datapoint(self.currentReference, None, None, source)
            self.model.initilize(temp_datapoint)
            self.calculated_profile = True
    def manual_update_and_reset(self,thresholdtype=None,thresholdfactor=None):
        if thresholdtype!=None:
            self.thresholdtype = thresholdtype
        if thresholdfactor!=None:
            self.thresholdfactor = thresholdfactor
        self.model = xgboostPerFeature.PairXgboost(thresholdtype, thresholdfactor,self.ActualProfileSize,constThresholdfilter=self.constThresholdfilter,actualProfileSize=self.alarmsThreshold)
    def get_current_reference(self):
        return self.currentReference

    def calculateReferenceData(self):
        if len(self.profilebuffer)>self.ProfileSize:


            data = [[seqdata.current for seqdata in seqpoint] for seqpoint in self.profilebuffer]
            tempnumpy=[]
            for sequense in data:
                tempnumpy.append(np.array([np.array(corrs) for corrs in sequense]))
            tempnumpy=np.array(tempnumpy)
            foundprofile = tempnumpy
            self.calculated_profile = True
            self.currentReference = foundprofile
    def get_Datapoint(self,timestamp : pd.Timestamp,data : ndarray, source) -> PredictionPoint:
        temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)

        if source==self.source:
            self.pointbuffer.append(temp_datapoint)
            if len(self.pointbuffer) >= self.sequencesize:
                self.pointbuffer = self.pointbuffer[-self.sequencesize:]


            if len(self.pointbuffer) < self.sequencesize:
                prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                             temp_datapoint.source, description="no profile yet")

            elif self.calculated_profile==False:
                # datapoint pass reference domain profile
                self.profilebuffer.append([dd for dd in self.pointbuffer])
                self.profilebuffertimestamps.append(timestamp)
                self.calculateReferenceData()
                if self.calculated_profile:
                    temp_datapoint = Datapoint(self.currentReference, np.array([np.array(corrs) for corrs in self.pointbuffer]), timestamp, source)
                    self.model.initilize(temp_datapoint)
                prediction= PredictionPoint(None, None, None,self.thresholdtype, temp_datapoint.timestamp,temp_datapoint.source,description="no profile yet")
            else:
                temp_datapoint = Datapoint(self.currentReference,
                                           np.array([np.array(corrs.current) for corrs in self.pointbuffer]), timestamp,
                                           source)
                prediction=self.model.get_data(temp_datapoint)
                prediction.description="xgboostpair detection"
            return prediction,temp_datapoint
        prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                     temp_datapoint.source, description="wrong source")
        return prediction,temp_datapoint

    def get_event(self, event: Eventpoint):
        for ev in self.resetcodes:
            if ev[0] == event.code and ev[1] == event.source:
                self.reset()
                return True, None, None
        return False, None, None

    def reset(self):
        self.profilebuffer = []
        self.profilebuffertimestamps = []
        self.calculated_profile = False
        self.currentReference = None
        self.model.reset()