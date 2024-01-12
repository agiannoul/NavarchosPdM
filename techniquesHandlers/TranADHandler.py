import pandas as pd
from numpy import ndarray

from Techniques import tranAD
from utils import distances as distcalc
from utils.structure import Datapoint, PredictionPoint, Eventpoint
def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

@auto_str
class tranADmethod():
    def __init__(self,source,window_size=10,num_epochs=15,thresholdtype="selftunne",thresholdfactor=2,metric="euclidean",ProfileSize=30,resetcodes=[],domainReference=None,domainRefFactor=4,currentReference=None,Auto_transform=False):
        self.profilebuffer=[]
        self.profilebuffertimestamps=[]
        self.maxinnerdistanceBuffer=[]
        self.datapointsBuffer=[]

        self.num_epochs = num_epochs
        self.window_size = window_size

        self.calculated_profile=False
        self.domainReference=domainReference
        self.domainRefFactor=domainRefFactor
        self.domainRefThreshold=None

        self.currentReference=currentReference
        self.resetcodes = resetcodes
        self.thresholdtype = thresholdtype
        self.thresholdfactor = thresholdfactor
        self.metric = metric
        self.source=source
        self.ProfileSize=ProfileSize
        self.Auto_transform=Auto_transform


        if self.domainReference is not None:
            self.domainRefThreshold=self.Maxporfiledist(self.domainReference,self.domainReference)*self.domainRefFactor

        self.model = tranAD.TranADPdm(num_epochs=self.num_epochs, window_size=self.window_size, source=self.source,dynamicThBufferSize=1,thresholdfactor=self.thresholdfactor,thresholdtype=self.thresholdtype,Auto_transform=self.Auto_transform)


        if self.currentReference is None:
            self.calculated_profile = False
        else:
            temp_datapoint = Datapoint(self.currentReference, None, None, source)
            self.model.initilize(temp_datapoint)
            self.calculated_profile = True



    def checkDomainReference(self,data):
        if self.domainReference is not None:
            distances=distcalc.calculate_distance_many_to_one(self.domainReference,data,self.metric)
            disterr=min(distances)
            if disterr<self.domainRefThreshold:
                return True
            else:
                return False
        else:
            return True

    def get_current_reference(self):
        return self.currentReference
    # get data and return PredictionPoint
    def get_Datapoint(self,timestamp : pd.Timestamp,data : ndarray, source) -> PredictionPoint:
        temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
        if source==self.source:
            if self.calculated_profile==False:
                temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
                self.datapointsBuffer.append(temp_datapoint)
                self.datapointsBuffer=self.datapointsBuffer[-self.window_size:]

                if self.checkDomainReference(data):
                    # datapoint pass reference domain profile
                    self.profilebuffer.append(data)
                    self.profilebuffertimestamps.append(timestamp)
                    self.calculateReferenceData()
                    temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
                    if self.calculated_profile:
                        self.model.initilize(temp_datapoint)
                    prediction= PredictionPoint(None, None, None,self.thresholdtype, temp_datapoint.timestamp,temp_datapoint.source,description="no profile yet")
                else:
                    #datapoint couldnot pass reference
                    prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                                 temp_datapoint.source,description="domain reference exclude")
            else:
                temp_datapoint = Datapoint(self.currentReference, data, timestamp, source)
                self.datapointsBuffer.append(temp_datapoint)
                self.datapointsBuffer = self.datapointsBuffer[-self.window_size:]
                prediction=self.model.get_data(self.datapointsBuffer[-self.window_size:])

            return prediction,temp_datapoint
        prediction = PredictionPoint(None, None, None, self.thresholdtype, temp_datapoint.timestamp,
                                     temp_datapoint.source, description="wrong source")
        return prediction,temp_datapoint

    def calculateReferenceData(self):
        if len(self.profilebuffer)>self.ProfileSize:
            tempdf=pd.DataFrame(self.profilebuffer[-self.ProfileSize:],index=self.profilebuffertimestamps[-self.ProfileSize:])
            self.maxinnerdistanceBuffer.append(self.Maxporfiledist(tempdf,tempdf))

            if len(self.maxinnerdistanceBuffer)>1:
                if self.maxinnerdistanceBuffer[-1]>self.maxinnerdistanceBuffer[-2]:
                    foundprofile = pd.DataFrame(self.profilebuffer[-self.ProfileSize-1:-1],
                                          index=self.profilebuffertimestamps[-self.ProfileSize-1:-1])
                    self.calculated_profile=True
                    self.currentReference=foundprofile

    def get_event(self,event: Eventpoint):
        for ev in self.resetcodes:
            if ev[0] == event.code and ev[1] == event.source:
                self.reset()
                #print("reset")
                return True, None, None
        return False, None, None
    def reset(self):
        self.profilebuffer = []
        self.profilebuffertimestamps = []
        self.maxinnerdistanceBuffer = []
        self.calculated_profile = False
        self.currentReference = None
        self.model.reset()
    def Maxporfiledist(self,df1,df2):
        distances = distcalc.calculate_distance_many_to_many(df1,df2,self.metric)
        maxdistProfile=[]
        for ar in distances:
            maxdistProfile.append(max(ar))

        return max(maxdistProfile)