import utils.transformation
from techniquesHandlers import pairdetectionHandler as pairHandler
from techniquesHandlers import TranADHandler as tranadh
from techniquesHandlers.grandInductiveHandler import grandInductiveHandler
from techniquesHandlers.xgboostPairHandler import XgboostPairHandler
from utils.aggregators import SmoothPredictionError
from utils.structure import modeltemp

def modelsnavinit(resetCodes, source):
    models = []

    smoother = SmoothPredictionError(size=3, smooth="median")

    ## Uncomment this to run closets Grand (inductive)
    # modelcur = grandInductiveHandler(source, resetcodes=resetCodes,non_comformity = "lof", k = 10,ProfileSize=3200,threshold=0.8,optimal=False)
    # name="grand inductive"

    # Uncomment this to run closets XGBoost
    ## Profile size is the length of Reference data
    ## ActualProfileSize : is what porpotion of the Reference should be used for Reference and threshold calculation
    ## thresholdfactor : the threshold factor used
    ## alarmsThreshold: in case we want to alarm only if there is violation on at least alarmsThreshold+1 of the features.
    #modelcur= XgboostPairHandler(source,thresholdtype="selftunne",thresholdfactor=14.5,ProfileSize=40,ActualProfileSize=0.70,resetcodes=resetCodes,alarmsThreshold=0)
    #name="xgboostpair detection"

    ## Uncomment this to run closets TranAD
    #modelcur=tranadh.tranADmethod(source, window_size=10, num_epochs=15, thresholdtype="selftunne", thresholdfactor=7, metric="euclidean", ProfileSize=3200, resetcodes=resetCodes, Auto_transform=True, dynamicThBufferSize=0)
    #name="tranadpdm"


    ## Uncomment this to run closets Pair Detection
    ## Profile size is the length of Reference data
    ## ActualProfileSize : is what porpotion of the Reference should be used for Reference and threshold calculation
    ## sequencesize: Choice if we want to considere subsequense sample instead of only the last element
    ## normz: True if we want znorm distance between samples else eucclidean is used
    ## thresholdfactor : the threshold factor used
    ## alarmsThreshold: in case we want to alarm only if there is violation on at least alarmsThreshold+1 of the features.
    modelcur = pairHandler.PairDectionHandler(source, thresholdtype="selftunne", thresholdfactor=15,
                                                        ProfileSize=30, ActualProfileSize=0.25, resetcodes=resetCodes,sequencesize=12,
                                                        alarmsThreshold=0,normz=False)
    name="pair detection"
    modeltorun = modeltemp(model=modelcur, source=source, name=name, needReference=True,
                         aggregations=[],transformations=[])

    # dtcEval = eventEvaluation.eventEvaluation(source,eventTargets=['storred','pending'])
    #
    # dtcdet = modeltemp(model=dtcEval,source=source,name="event prediction",needReference=False,iseventgenrated=True)


    models.append(modeltorun)

    ensembles = []
    return models, ensembles