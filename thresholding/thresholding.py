import statistics




# calculates a threshodl based on scores derived from relative normal samples.
# The threshold is equal to the mean of such scores plus their standard deviation multiplied by a factor
def selfTuning(factor,anomalyscoresInNormal=[],anomalyscores=[],sizeOfReference=None,returnmean=False):
    if len(anomalyscoresInNormal)==0:
        if len(anomalyscores)<sizeOfReference:
            return False,max(anomalyscores) # not enough data to calculate threshold using the parameters.
        anomalyscoresInNormal=anomalyscores[:sizeOfReference]
    thmean=statistics.mean(anomalyscoresInNormal)
    thstd=statistics.stdev(anomalyscoresInNormal)
    th = thmean + factor * thstd
    if returnmean:
        return th,thmean,thstd
    return th


