import pandas as pd
from numpy import ndarray
from utils import distances as distcalc

from Techniques.grandInductive import grandInductive_core
from utils.structure import PredictionPoint, Datapoint, Eventpoint

def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

@auto_str
class grandInductiveHandler():

    def __init__(self,source,resetcodes = [],non_comformity = "knn", k = 20,ProfileSize=30,threshold=0.5,optimal=False):
        self.non_comformity =non_comformity
        self.k = k
        self.source = source
        self.threshold =threshold
        self.method = grandInductive_core(self.non_comformity,self.k)
        self.currentReference=None
        self.calculated_profile = False
        self.profilebuffer=[]
        self.profilebuffertimestamps=[]
        self.maxinnerdistanceBuffer=[]
        self.ProfileSize=ProfileSize
        self.optimalProfile=optimal
        self.thresholdtype = "constant"
        self.resetcodes=resetcodes


    def get_current_reference(self):
        return self.currentReference

    def calculateReferenceData(self):
        if len(self.profilebuffer)>self.ProfileSize:
            tempdf=pd.DataFrame(self.profilebuffer[-self.ProfileSize:],index=self.profilebuffertimestamps[-self.ProfileSize:])

            if self.optimalProfile:
                self.maxinnerdistanceBuffer.append(self.Maxporfiledist(tempdf, tempdf))
                if len(self.maxinnerdistanceBuffer)>1:
                    if self.maxinnerdistanceBuffer[-1]>self.maxinnerdistanceBuffer[-2]:
                        foundprofile = pd.DataFrame(self.profilebuffer[-self.ProfileSize-1:-1],
                                              index=self.profilebuffertimestamps[-self.ProfileSize-1:-1])
                        self.calculated_profile=True
                        self.currentReference=foundprofile
            else:
                foundprofile=tempdf
                self.calculated_profile = True
                self.currentReference = foundprofile

    def get_Datapoint(self, timestamp: pd.Timestamp, data: ndarray, source) -> PredictionPoint:
        temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
        if source==self.source:
            if self.calculated_profile==False:
                self.profilebuffer.append(data)
                self.profilebuffertimestamps.append(timestamp)
                self.calculateReferenceData()
                if self.calculated_profile:
                    temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
                    self.method = grandInductive_core(self.non_comformity,self.k)
                    self.method.fit(self.currentReference.values)
                prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                         temp_datapoint.source, description="no profile yet")
            else:
                prediction = self.perform_prediction(temp_datapoint)

            return prediction,temp_datapoint
        prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                         temp_datapoint.source, description="wrong source")
        return prediction, temp_datapoint


    def perform_prediction(self,datapoint : Datapoint):
        deviation = self.method.predict(datapoint.timestamp,datapoint.current)
        prediction = PredictionPoint(deviation, self.threshold, deviation>self.threshold, self.thresholdtype, datapoint.timestamp,
                                     datapoint.source, description="grand inductive")
        return prediction


    def get_event(self,event: Eventpoint):
        for ev in self.resetcodes:
            if ev[0] == event.code and ev[1] == event.source:
                self.reset()
                return True,None,None
        return False,None,None
    def reset(self):
        self.profilebuffer = []
        self.profilebuffertimestamps = []
        self.maxinnerdistanceBuffer = []
        self.calculated_profile = False
        self.currentReference = None
        self.method = grandInductive_core(self.non_comformity,self.k)



    def Maxporfiledist(self,df1,df2):
        distances = distcalc.calculate_distance_many_to_many(df1,df2,self.metric)
        maxdistProfile=[]
        for ar in distances:
            maxdistProfile.append(max(ar))

        return max(maxdistProfile)