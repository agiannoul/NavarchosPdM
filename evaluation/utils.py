import matplotlib.pyplot as plt
import pandas as pd
import evaluation.evalution as eval


#
def finaleval(allalarms, allfailuredates, alldates,PH="14 days",lead="1 hours",plotThem=False):
    allepisodes, allepisodesdates, allisfailure = evalApproach(allalarms, allfailuredates, alldates)
    for ep, epdate in zip(allepisodes, allepisodesdates):
        print(f"{len(ep)} - {len(epdate)}")
    recall, Precision, f1, axes = eval.myeval(allepisodes, 0.5, datesofscores=allepisodesdates, PH=PH,
                                              lead=lead, isfailure=allisfailure,plotThem=plotThem,beta=2)
    if plotThem:
        print(Precision)
        print(recall)
        print(f1)
        plt.show()
    return  f1[0]
def evalApproach(allalarms, allfailuredates, alldates):
    allepisodes = []
    allepisodesdates = []
    allisfailure = []

    for alarms, failuredates, dates in zip(allalarms, allfailuredates, alldates):
        isfailure,episodes, episodesdates = brealIntoEpisodes(alarms, failuredates, dates)
        allepisodes.extend(episodes)
        allepisodesdates.extend(episodesdates)
        allisfailure.extend(isfailure)
    return allepisodes,allepisodesdates,allisfailure
def brealIntoEpisodes(alarms,failuredates,thresholds,dates):
    isfailure=[]
    episodes=[]
    episodesthreshold=[]
    episodesdates=[]
    #dates=[pd.to_datetime(datedd) for datedd in dates]
    #failuredates=[pd.to_datetime(datedd) for datedd in failuredates]


    # no failures
    if len(failuredates)==0:
        if len(alarms)>0:
            isfailure.append(0)
            episodes.append(alarms)
            episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure,episodes,episodesdates,episodesthreshold

    counter=0
    for fdate in failuredates:
        for i in range(counter,len(dates)):
            if dates[i]>fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    episodes.append(alarms[counter:i])
                    episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter=i
                break
    if dates[-1]<failuredates[-1]:
        isfailure.append(1)
        episodes.append(alarms[counter:])
        episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter<len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            episodes.append(alarms[counter:])
            episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure,episodes, episodesdates,episodesthreshold


def brealIntoEpisodesWithCodes(alarms,failuredates,failurecodes,thresholds,dates):
    isfailure=[]
    failuretype=[]
    episodes=[]
    episodesthreshold=[]
    episodesdates=[]
    #dates=[pd.to_datetime(datedd) for datedd in dates]
    #failuredates=[pd.to_datetime(datedd) for datedd in failuredates]


    # no failures
    if len(failuredates)==0 or len(dates)==0:
        if len(alarms)>0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms)
            episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure,episodes,episodesdates,episodesthreshold,failuretype

    counter=0
    for fdate,ftype in zip(failuredates,failurecodes):
        for i in range(counter,len(dates)):
            if dates[i]>fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    failuretype.append(ftype)
                    episodes.append(alarms[counter:i])
                    episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter=i
                break
    if dates[-1]<failuredates[-1]:
        isfailure.append(1)
        failuretype.append(failurecodes[-1])
        episodes.append(alarms[counter:])
        episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter<len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms[counter:])
            episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure,episodes, episodesdates,episodesthreshold,failuretype