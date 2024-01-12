import pandas as pd
from pandas import DataFrame
from numpy import ndarray

class modeltemp():
    def __init__(self, model, source, name,isbatch=False,needReference=False,transformations=[],aggregations=[],uid=1,iseventgenrated=False):
        self.model=model
        self.uid=uid
        self.source=source
        self.name=name
        self.needReference=needReference
        self.predictions=[]
        self.isbatch=isbatch
        self.iseventgenrated=iseventgenrated
        self.transformations=transformations
        self.aggregations=aggregations
        self.aggregatedPredictions=[]
        for aggregator in self.aggregations:
            self.aggregatedPredictions.append([])


    def get_data(self,dttime,row,source,desc):
        if self.source!=source:
            pred=PredictionPoint(None,None,None,None,None,None,description="wronk source")
            return pred,None
        for transf in self.transformations:
            row=transf.transform(row)
        if self.isbatch:
            prediction,cdatapoint = self.model.get_Datapoint(dttime, row, source,desc)
            # convert batch to point predictions
            self.batchtoPointPredictions_andAppend( prediction)
        else:
            if self.name=="xgboostStemp":
                prediction, cdatapoint = self.model.get_Datapoint(dttime, row, source,desc)
            else:
                prediction,cdatapoint = self.model.get_Datapoint(dttime, row, source)
            self.append_to_predictions(prediction)
        # if aggregator is used return aggregator value
        if len(self.aggregations):
            if self.isbatch==False:
                aggregatedpred=self.aggregations[0].transform(self.predictions,prediction)
                self.aggregatedPredictions[0].append(aggregatedpred)
                return aggregatedpred,cdatapoint
            else:
                predictedpoints=len(prediction.score)
                aggrscorestemp=[]
                aggrthresholdtemp=[]
                aggrlastshifttimestemp=[]
                aggralarmtemp=[]
                for i in range(predictedpoints):
                    aggregatedpred = self.aggregations[0].transform(self.predictions[:-(predictedpoints-i)], self.predictions[-(predictedpoints-i)])
                    aggrscorestemp.append(aggregatedpred.score)
                    aggralarmtemp.append(aggregatedpred.alarm)
                    aggrthresholdtemp.append(aggregatedpred.threshold)
                    aggrlastshifttimestemp.append(aggregatedpred.timestamp)
                batchpredtemp=BatchPredictionPoint(score=aggrscorestemp, threshold=aggrthresholdtemp,
                                         alarm=aggralarmtemp,
                                         timestamp=aggrlastshifttimestemp, source=prediction.source, description=prediction.description)

                return batchpredtemp,cdatapoint
        return prediction, cdatapoint
    def batchtoPointPredictions_andAppend(self,prediction):
        if prediction.score is None:
            temppred = PredictionPoint(score=None, timestamp=None, thresholdtype=None, alarm=None,
                                       threshold=prediction.threshold, source=prediction.source,
                                       description=prediction.description)

            self.append_to_predictions(temppred)
        else:
            for sc, tm, alrm in zip(prediction.score, prediction.timestamp, prediction.alarm):
                temppred = PredictionPoint(score=sc, timestamp=tm, thresholdtype="batch", alarm=alrm,
                                           threshold=prediction.threshold, source=prediction.source,
                                           description=prediction.description)
                self.append_to_predictions(temppred)
    def get_event(self,eventpoint):
        if self.isbatch:
            resetted,prediction,cdatapointlist=self.model.get_event(eventpoint)
            if resetted:
                self.batchtoPointPredictions_andAppend(prediction)
                return prediction,cdatapointlist
            return None,None
        elif self.iseventgenrated:
            prediction=self.model.get_event(eventpoint)
            if prediction is not None:
                self.append_to_predictions(prediction)
            return prediction,None
        else:
            resetted,prediction,cdatapoint=self.model.get_event(eventpoint)
            return prediction,cdatapoint
    def get_reference(self):
        if self.needReference:
            return self.model.get_current_reference()
        else:
            return False

    def append_to_predictions(self,temppred):
        self.predictions.append(temppred)
        count=-1
        for aggregator in self.aggregations:
            count+=1
            aggregatedpred=aggregator.transform(self.predictions,temppred)
            self.aggregatedPredictions[count].append(aggregatedpred)
class PredictionPoint():
    def __init__(self, score, threshold, alarm, thresholdtype, timestamp, source, description="",notes="",ensemble_details=None):
        self.score = score
        self.threshold = threshold
        self.alarm = alarm
        self.thresholdtype = thresholdtype
        self.timestamp = timestamp
        self.source = source
        self.description = description
        self.notes = notes
        self.ensemble_details = ensemble_details

    def toString(self):
        kati=f"{self.source},{self.timestamp},{self.score},{self.threshold},{self.thresholdtype},{self.notes}"
        if kati[-1] == ",":
            kati= kati[:-1]
        return kati

class BatchPredictionPoint():
    def __init__(self, score, threshold, alarm, timestamp, source,thresholdtype=None, description="",ensemble_details=None):
        self.score = score
        self.threshold = threshold
        self.thresholdtype = thresholdtype
        self.alarm = alarm
        self.timestamp = timestamp
        self.source = source
        self.description = description
        self.ensemble_details = ensemble_details

class Datapoint():

    def __init__(self,reference :DataFrame,current: ndarray,timestamp : pd.Timestamp,source):
        self.reference = reference
        self.current = current
        self.source = source
        self.timestamp = timestamp

class SimpleDatapoint():

    def __init__(self,current: ndarray,timestamp : pd.Timestamp,source):
        self.current = current
        self.source = source
        self.timestamp = timestamp
        self.description = None

class Eventpoint():

    def __init__(self,code,source,timestamp,details=None,type=None):
        self.code = code
        self.source = source
        self.timestamp = timestamp
        self.details = details
        self.type = type