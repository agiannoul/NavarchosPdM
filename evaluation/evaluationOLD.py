import collections

from prts import ts_recall
import pandas as pd
import matplotlib.pyplot as plt
import math

# This method is used to perform PdM evaluation of Run-to-Failures examples.
# predictions: Either a flatted list of all predictions from all episodes or
#               list with a list of prediction for each of episodes
# datesofscores: Either a flatted list of all indexes (timestamps) from all episodes or
#                list with a list of  indexes (timestamps) for each of episodes
#                If it is empty list then aritificially indexes are gereated
# threshold: can be either a list of thresholds (equal size to all predictions), a list with size equal to number of episodes, a single number.
# maintenances: is used in case the predictions are passed as flatten array (default None)
#   list of ints which indicate the time of maintenance (the position in predictions where a new episode begins) or the end of the episode.
# isfailure: a binary array which is used in case we want to pass episode which end with no failure, and thus don't contribute
#   to recall calculation. For example isfailure=[1,1,0,1] indicates that the third episode end with no failure, while the others end with a failure.
#   default value is empty list which indicate that all episodes end with failure.
# PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
#   ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
#   in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
#   are used for predictive horizon
#lead: represent the lead time (the time to ignore in last part of episode when we want to test the predict capabilities of algorithm)
#   same rules as the PH are applied.
# ignoredates: a list with tuples in form (begin,end) which indicate periods to be ignored in the calculation of recall and precision
#   begin and end values must be same type with datesofscores instances (pd.datetime or int)
# beta is used to calculate fbeta score deafult beta=1.
def myeval(predictions,threshold,datesofscores=[],maintenances=None,isfailure=[],PH="100",lead="20",plotThem=True,ignoredates=[],beta=1):



    artificialindexes=[]
    thresholdtempperepisode=[]
    if isinstance(predictions[0], collections.abc.Sequence):
        temppreds=[]
        maintenances=[]
        if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(predictions):
            for episodepreds,thepisode in zip(predictions,threshold):
                thresholdtempperepisode.extend([thepisode for i in range(len(episodepreds))])
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i+len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        else:
            for episodepreds in predictions:
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i+len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        predictions=temppreds


    if len(datesofscores) == 0:
        datesofscores=artificialindexes
    elif isinstance(datesofscores[0], collections.abc.Sequence):
        temppreds=[]
        for episodeindexesss in datesofscores:
            temppreds.extend(episodeindexesss)
        datesofscores=temppreds
    if maintenances == None:
        assert False,"When you pass a flatten array for predictions, maintenances must be assigned to cutoffs time/indexes"
    if maintenances[-1] != len(predictions):
        assert False,"The maintenance indexes are not alligned with predictions length (last index of predictions should be the last element of maintenances)"
    if len(predictions) != len(datesofscores):
        assert False,f"Inconsistency in the size of scores (predictions) and dates-indexes {len(predictions)} != {len(datesofscores)}"


    if len(isfailure)==0:
        isfailure=[1 for m in maintenances]




    if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(maintenances):
        threshold=thresholdtempperepisode
    elif isinstance(threshold, collections.abc.Sequence)==False:
        temp=[threshold for i in predictions]
        threshold=temp

    assert len(predictions) == len(threshold), f"Inconsistency in the size of scores (predictions {len(predictions)}) and thresholds {len(threshold)}"



    if len(PH.split(" "))<2:
        numbertime = int(PH.split(" ")[0])
        timestyle = ""
    else:
        scale = PH.split(" ")[1]
        acceptedvalues=["","days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertime = int(PH.split(" ")[0])

            timestyle=scale
        else:
            assert False,f"PH parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    if len(lead.split(" "))<2:
        numbertimelead = int(lead.split(" ")[0])
        timestylelead = ""
    else:
        scale = lead.split(" ")[1]
        if scale in acceptedvalues:
            numbertimelead = int(lead.split(" ")[0])

            timestylelead = scale
        else:
            assert False,f"lead parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"


    anomalyranges = [0 for i in range(maintenances[-1])]



    prelimit = 0
    totaltp, totalfp, totaltn, totalfn = 0, 0, 0, 0
    arraytoplotall, toplotbordersall = [],[]
    counter=-1
    thtoreturn=[]
    episodescounts=len(maintenances)
    axes=None
    if plotThem:
        fig, axes = plt.subplots(nrows=math.ceil(episodescounts / 4), ncols=min(4,len(maintenances)))
        if math.ceil(episodescounts / 4)==1:
            emptylist=[]
            emptylist.append(axes)
            axes=emptylist
    predictionsforrecall=[]
    countaxes=0

    for maint in maintenances:

        counter+=1
        if isfailure[counter]==1:

            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            episodealarms = [v > th for v, th in zip(episodePred, episodethreshold)]
            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])
            tempdatesofscores = datesofscores[prelimit:maint]



            tp, fp, borderph, border1 = episodeMyPresicionTPFP(episodealarms, tempdatesofscores,PredictiveHorizon=numbertime,leadtime=numbertimelead,timestylelead=timestylelead,timestyle=timestyle,ignoredates=ignoredates)
            if counter > 0:
                for i in range(borderph+maintenances[counter-1], border1+maintenances[counter-1]):
                    anomalyranges[i] = 1
            else:
                for i in range(borderph, border1 ):
                    anomalyranges[i] = 1
            totaltp += tp
            totalfp += fp
            prelimit = maint

            if plotThem:
                if timestyle=="":
                    for i in range(len(ignoredates)):
                        if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                            pos1 = -1
                            pos2 = -1
                            for q in range(len(tempdatesofscores)):
                                if tempdatesofscores[q] > ignoredates[i][0]:
                                    pos1 = q
                                    break
                            for q in range(pos1, len(tempdatesofscores)):
                                if tempdatesofscores[q] > ignoredates[i][1]:
                                    pos2 = q
                                    break
                            # print(
                            #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                            # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                            axes[countaxes // 4][countaxes % 4].fill_between(
                                tempdatesofscores[pos1:pos2], max(max(episodethreshold),max(episodePred)),
                                min(episodePred),
                                color="grey",
                                alpha=0.3,
                                label="ignore")
                else:
                    for i in range(len(ignoredates)):
                        if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                                tempdatesofscores[-1].tz_localize(None):
                            pos1 = -1
                            pos2 = -1
                            for q in range(len(tempdatesofscores)):
                                if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                                    pos1 = q
                                    break
                            for q in range(pos1, len(tempdatesofscores)):
                                if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                                    pos2 = q
                                    break
                            # print(
                            #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                            # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                            axes[countaxes // 4][countaxes % 4].fill_between(
                                tempdatesofscores[pos1:pos2], max(max(episodethreshold),max(episodePred)),
                                min(episodePred),
                                color="grey",
                                alpha=0.3,
                                label="ignore")

                #============================================================

                axes[countaxes//4][countaxes%4].plot(tempdatesofscores,episodePred, label="pb")
                axes[countaxes//4][countaxes%4].fill_between([tempdatesofscores[i] for i in range(borderph, border1)], max(max(episodethreshold),max(episodePred)),
                                 min(episodePred), where=[1 for i in range(borderph, border1)], color="red",
                                 alpha=0.3,
                                 label="PH")
                axes[countaxes//4][countaxes%4].fill_between([tempdatesofscores[i] for i in range(border1, len(episodePred))], max(max(episodethreshold),max(episodePred)),
                                             min(episodePred), where=[1 for i in range(border1, len(episodePred))],
                                             color="grey",
                                             alpha=0.3,
                                             label="ignore")

                axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores,episodethreshold, color="k", linestyle="--", label="th")

                #axes[countaxes//4][countaxes%4].legend()

                countaxes+=1
        else:


            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])


            tempdatesofscores = datesofscores[prelimit:maint]
            if timestyle != "":
                for score,th,datevalue in zip(episodePred,episodethreshold,tempdatesofscores):
                    if ignore(datevalue,ignoredates):
                        if score>th:
                            totalfp+=1
            else:
                for score,th,datevalue in zip(episodePred,episodethreshold,tempdatesofscores):
                    if ingnorecounter(datevalue,ignoredates):
                        if score>th:
                            totalfp+=1
            if plotThem:

                if timestyle == "":
                    for i in range(len(ignoredates)):
                        if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                            pos1 = -1
                            pos2 = -1
                            for q in range(len(tempdatesofscores)):
                                if tempdatesofscores[q] > ignoredates[i][0]:
                                    pos1 = q
                                    break
                            for q in range(pos1, len(tempdatesofscores)):
                                if tempdatesofscores[q] > ignoredates[i][1]:
                                    pos2 = q
                                    break
                            # print(
                            #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                            # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                            axes[countaxes // 4][countaxes % 4].fill_between(
                                tempdatesofscores[pos1:pos2], max(max(episodethreshold),max(episodePred)),
                                min(episodePred),
                                color="grey",
                                alpha=0.3,
                                label="ignore")
                else:
                    for i in range(len(ignoredates)):
                        if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                                tempdatesofscores[-1].tz_localize(None):
                            pos1 = -1
                            pos2 = -1
                            for q in range(len(tempdatesofscores)):
                                if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                                    pos1 = q
                                    break
                            for q in range(pos1, len(tempdatesofscores)):
                                if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                                    pos2 = q
                                    break
                            # print(
                            #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                            # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                            axes[countaxes // 4][countaxes % 4].fill_between(
                                tempdatesofscores[pos1:pos2], max(max(episodethreshold),max(episodePred)),
                                min(episodePred),
                                color="grey",
                                alpha=0.3,
                                label="ignore")
                # ============================================================




                axes[countaxes//4][countaxes%4].plot(tempdatesofscores,episodePred,color="green", label="pb n")
                axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores,episodethreshold, color="k", linestyle="--", label="th")
                #axes[countaxes//4][countaxes%4].legend()
                countaxes+=1
            prelimit = maint
        #print(f"Counter {counter} predslen: {len(predictionsforrecall)}, anomalyrangeslen:{len(anomalyranges)}" )
    if sum(predictionsforrecall)==0:
        AD1=0
        AD2=0
        AD3=0
    else:
        AD1 = ts_recall(anomalyranges, predictionsforrecall, alpha=1, cardinality="one", bias="flat")
        AD2 = ts_recall(anomalyranges, predictionsforrecall, alpha=0, cardinality="one", bias="flat")
        AD3 = ts_recall(anomalyranges, predictionsforrecall, alpha=0, cardinality="one", bias="back")
        AD3 = AD3 * AD2

    Precision = 0
    if totaltp+totalfp!=0:
        Precision=totaltp/(totaltp+totalfp)
    recall=[AD1,AD2,AD3]
    f1=[]
    for rec in recall:
        if Precision+rec==0:
            f1.append(0)
        else:
            F = ((1 + beta ** 2) * Precision * rec) / (beta ** 2 * Precision + rec)
            f1.append(F)
    # if plotThem:
    #     print("=====================")
    #     print(recall)
    #     print(Precision)
    #     print(f1)
    #     print("=====================")
    #     #plt.show()


    return recall,Precision,f1,axes


def episodeMyPresicionTPFP(episodealarms,tempdatesofscores,PredictiveHorizon,leadtime,timestyle,timestylelead,ignoredates):
    border2=len(episodealarms)
    totaltp=0
    totalfp=0

    arraytoplot=[]
    toplotborders=[]


    border2date = tempdatesofscores[border2 - 1]

    border1 = border2 - 1
    if timestyle!="":
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(leadtime,timestylelead):
                border1 = i -1
                if border1==-1:
                    border1=0
                break
        borderph=border1-1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(PredictiveHorizon,timestyle):
                borderph = i - 1
                if borderph==-1:
                    borderph=0
                break
        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ignore(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1

    else:
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - leadtime:
                border1 = i - 1
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - PredictiveHorizon:
                borderph = i - 1
                break

        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ingnorecounter(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1


def calculatePHandLead(PH,lead):
    if len(PH.split(" "))<2:
        numbertime = int(PH.split(" ")[0])
        timestyle = ""
    else:
        scale = PH.split(" ")[1]
        acceptedvalues=["","days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertime = int(PH.split(" ")[0])

            timestyle=scale
        else:
            assert False,f"PH parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    if len(lead.split(" "))<2:
        numbertimelead = int(lead.split(" ")[0])
        timestylelead = ""
    else:
        scale = lead.split(" ")[1]
        if scale in acceptedvalues:
            numbertimelead = int(lead.split(" ")[0])

            timestylelead = scale
        else:
            assert False,f"lead parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    return numbertime,timestyle,numbertimelead,timestylelead

def ignore(valuedate,tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate.tz_localize(None)>tup[0] and valuedate.tz_localize(None)<tup[1]:
            return False
    return True
def ingnorecounter(valuedate,tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate>tup[0] and valuedate<tup[1]:
            return False
    return True