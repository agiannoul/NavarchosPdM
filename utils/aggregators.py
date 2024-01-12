from utils.structure import PredictionPoint
import statistics


class SmoothPredictionError():

    def __init__(self,size=30,smooth="median",threshold=None):
        self.size=size
        self.smooth=smooth
        self.threshold=threshold
    def transform(self,listofPrection: list[PredictionPoint],lastpred: PredictionPoint) -> PredictionPoint:
        if lastpred.score is None:
            return lastpred
        noNonelist=[pr for pr in listofPrection if pr.score is not None]
        noNonelist=noNonelist[-self.size:]

        noNonelistscore=[pr.score for pr in noNonelist]
        aggregatedscore = lastpred.score
        if self.smooth == "mean":
            aggregatedscore = sum(noNonelistscore)/len(noNonelistscore)
        elif self.smooth == "median":
            aggregatedscore = statistics.median(noNonelistscore)
        else:
            aggregatedscore = lastpred.score
        if self.threshold is None:
            predpoint = PredictionPoint(score=aggregatedscore, threshold=lastpred.threshold,
                                    alarm=aggregatedscore > lastpred.threshold, thresholdtype=lastpred.thresholdtype,
                                    timestamp=lastpred.timestamp, source=lastpred.source,
                                    description=lastpred.description)
        else:
            predpoint = PredictionPoint(score=aggregatedscore, threshold=self.threshold,
                                        alarm=aggregatedscore > self.threshold,
                                        thresholdtype="constant",
                                        timestamp=lastpred.timestamp, source=lastpred.source,
                                        description=lastpred.description)
        return predpoint