import statistics
import evaluation.utils as evalutils
import evaluation.evalution as eval
from matplotlib import pyplot as plt
from utils.structure import Eventpoint, SimpleDatapoint, modeltemp
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots



# fussion of models and ensebles with meta_techniqeus
class fuss():
    # ContextDataTypes takes values: timeseries and points
    def __init__(self, models, ensembles=[]):
        self.models = models
        self.ensembles = ensembles


    def bubbleSort(self, arr):
        n = len(arr)
        # optimize code, so if the array is already sorted, it doesn't need
        # to go through the entire process
        swapped = False
        # Traverse through all array elements
        for i in range(n - 1):
            # range(n) also work but outer loop will
            # repeat one time more than needed.
            # Last i elements are already in place
            for j in range(0, n - i - 1):

                # traverse the array from 0 to n-i-1
                # Swap if the element found is greater
                # than the next element
                if arr[j].timestamp > arr[j + 1].timestamp:
                    swapped = True
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

            if not swapped:
                # if we haven't needed to make a single swap, we
                # can just exit the main loop.
                return arr
        return arr

    def feedDataToModels(self, data, dttime, source, desc):

        for modelt in self.models:
            if modelt.source == source:
                prediction, cdatapoint = modelt.get_data(dttime, data, source, desc)

        for modelt in self.ensembles:
            if modelt.source == source:
                prediction = modelt.get_ensemble_score(dttime, source)


    def feedEventToModels(self, eventpoint: Eventpoint):
        for modelt in self.models:
            prediction, cdatapointlist = modelt.get_event(eventpoint)

    # for monitored data
    def collect_data(self, row, dttime, source, desc=""):
        self.feedDataToModels(row, dttime, source, desc)


    # for event data
    def collect_event(self, eventpoint: Eventpoint):
        self.feedEventToModels(eventpoint)

    def debug_forplot_score_analytical(self,listpreds,model):
        score = []
        timescore = []
        thresholds = []
        length=None
        for pred in listpreds:
            if pred.description == model.name:
                if pred.ensemble_details is None:
                    score.append(float(pred.score))
                    timescore.append(pred.timestamp)
                    thresholds.append(pred.threshold)
                else:
                    if length is None:
                        length=len(pred.ensemble_details[0])
                        for q in range(length):
                            score.append([])
                            timescore.append([])
                            thresholds.append([])
                    else:
                        for q in range(length):
                            score[q].append(pred.ensemble_details[1][q])
                            thresholds[q].append(pred.ensemble_details[0][q])
                            timescore[q].append(pred.timestamp)
        return score, timescore, thresholds


    def debug_plot_analytical(self,failures=[], failurecodes=[], eventsofint=[], failuresources=[], eventsofintsources=[],
                   eventcodes=[], matcColor=[]):
        if len(self.models) + len(self.ensembles) >= 10:
            subnumberplot = 910
        else:
            subnumberplot = (len(self.models) + len(self.ensembles)) * 100 + 10

        counter = 0
        for model in self.models:
            print(f"Model {model.name} source {model.source}")

            counter += 1
            score = []
            timescore = []
            thresholds = []
            if model.isbatch:
                score, timescore, thresholds = self.debug_forplot_score_analytical(model.predictions, model)
            else:
                score, timescore, thresholds = self.debug_forplot_score_analytical(model.predictions, model)
            if len(score)==0:
                continue
            if score[0] is not None and isinstance(score[0], list):
                gridx=4
                gridy=len(score)//gridx + 1
                fig, axes = plt.subplots(nrows=gridy, ncols=gridx)
                fig.suptitle(f"{model.name} source: {model.source}")
                allaxes=[]
                for qy in range(gridy):
                    for qx in range(gridx):
                        allaxes.append(axes[qy][qx])

                for qi in range(len(score)):
                    allaxes[qi].scatter(timescore[qi], score[qi])
                    allaxes[qi].plot(timescore[qi], thresholds[qi], ".-", color="pink")

                    # EVENTS
                    if len(eventcodes) == 0:
                        self.poltEvents(eventsofint, eventsofintsources, model, color="magenta",ax=allaxes[qi])
                    else:
                        self.poltEventsWithDesc(score[qi], eventsofint, eventcodes, eventsofintsources, model,
                                                color="magenta",
                                                matchcolors=matcColor, failuresplot=False,ax=allaxes[qi])

                    # FAILURES
                    if len(failurecodes) == 0:
                        # WITHOUT DESC
                        self.poltEvents(eventsofint, eventsofintsources, model, color="red",ax=allaxes[qi])
                    else:
                        # WITH DESC
                        self.poltEventsWithDesc(score[qi], failures, failurecodes, failuresources, model, color="red",ax=allaxes[qi])
                plt.show()
            else:
                plt.subplot(subnumberplot + counter)
                plt.title(f"{model.name} source: {model.source}")
                plt.scatter(timescore, score)

                # set ylim
                if len(score) > 2:
                    meanscore = statistics.mean(score)
                    maxscore = max(score)
                    minscore = min(score)
                    stdscore = statistics.stdev(score)
                    if stdscore == 0:
                        stdscore = 1
                    plt.ylim((min(minscore, meanscore - stdscore * 1.1), max(maxscore, meanscore + stdscore * 1.1)))

                for i in range(len(model.aggregations)):
                    score, timescore, thresholds = self.debug_forplot_score(model.aggregatedPredictions[i], model)
                    plt.plot(timescore, score, ".-", color="grey")
                plt.plot(timescore, thresholds, ".-", color="pink")

                # EVENTS
                if len(eventcodes) == 0:
                    self.poltEvents(eventsofint, eventsofintsources, model, color="magenta")
                else:
                    self.poltEventsWithDesc(score, eventsofint, eventcodes, eventsofintsources, model, color="magenta",
                                            matchcolors=matcColor, failuresplot=False)

                # FAILURES
                if len(failurecodes) == 0:
                    # WITHOUT DESC
                    self.poltEvents(eventsofint, eventsofintsources, model, color="red")
                else:
                    # WITH DESC
                    self.poltEventsWithDesc(score, failures, failurecodes, failuresources, model, color="red")

                if counter == 9:
                    counter = 0
                    plt.show()
    def write_results(self):
        for model in self.models:
            with open(f'results_{model.name}.resbig', 'a+') as the_file:
                for pred in model.predictions:
                    if pred.score != None:
                        the_file.write(pred.toString())
                        the_file.write("\n")
    def debug_plot(self, failures=[], failurecodes=[], eventsofint=[], failuresources=[], eventsofintsources=[],
                   eventcodes=[], matcColor=[]):

        if len(self.models) + len(self.ensembles) >= 10:
            subnumberplot = 910
        else:
            subnumberplot = (len(self.models) + len(self.ensembles)) * 100 + 10

        counter = 0
        for model in self.models:
            print(f"Model {model.name} source {model.source}")
            counter += 1
            score = []
            timescore = []
            thresholds = []
            if model.isbatch:
                score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            else:
                score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            plt.subplot(subnumberplot + counter)
            plt.title(f"{model.name} source: {model.source}")
            plt.scatter(timescore, score)

            # set ylim
            if len(score) > 2:
                meanscore = statistics.mean(score)
                maxscore = max(score)
                minscore = min(score)
                stdscore = statistics.stdev(score)
                if stdscore == 0:
                    stdscore = 1
                plt.ylim((min(minscore, meanscore - stdscore * 1.1), max(maxscore, meanscore + stdscore * 1.1)))

            for i in range(len(model.aggregations)):
                score, timescore, thresholds = self.debug_forplot_score(model.aggregatedPredictions[i], model)
                plt.plot(timescore, score, ".-", color="grey")
            plt.plot(timescore, thresholds, ".-", color="pink")

            # EVENTS
            if len(eventcodes) == 0:
                self.poltEvents(eventsofint, eventsofintsources, model, color="magenta")
            else:
                self.poltEventsWithDesc(score, eventsofint, eventcodes, eventsofintsources, model, color="magenta",
                                        matchcolors=matcColor, failuresplot=False)

            # FAILURES
            if len(failurecodes) == 0:
                # WITHOUT DESC
                self.poltEvents(eventsofint, eventsofintsources, model, color="red")
            else:
                # WITH DESC
                self.poltEventsWithDesc(score, failures, failurecodes, failuresources, model, color="red")

            if counter == 9:
                counter = 0
                plt.show()
        for model in self.ensembles:
            print(f"Model {model.name} source: {model.source}")
            counter += 1
            score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            plt.subplot(subnumberplot + counter)
            plt.title(f"{model.name} source: {model.source}")
            plt.scatter(timescore, score)
            plt.plot(timescore, thresholds, ".-", color="red")

            # set ylim
            if len(score) > 2:
                meanscore = statistics.mean(score)
                maxscore = max(score)
                minscore = min(score)
                stdscore = statistics.stdev(score)
                if stdscore == 0:
                    stdscore = 1
                plt.ylim((min(minscore, meanscore - stdscore * 1.1), max(maxscore, meanscore + stdscore * 1.1)))

            # EVENTS
            if len(eventcodes) == 0:
                self.poltEvents(eventsofint, eventsofintsources, model, color="magenta")
            else:
                self.poltEventsWithDesc(score, eventsofint, eventcodes, eventsofintsources, model, color="magenta",
                                        matchcolors=matcColor, failuresplot=False)

            # FAILURES
            if len(failurecodes) == 0:
                # WITHOUT DESC
                self.poltEvents(eventsofint, eventsofintsources, model, color="red")
            else:
                # WITH DESC
                self.poltEventsWithDesc(score, failures, failurecodes, failuresources, model, color="red",
                                        failuresplot=True)

        plt.savefig('./images/tempresults.png')
        plt.show()

    def poltEvents(self, eventsofint, eventsofintsources, model, color="red",ax=None):
        if ax is None:
            for fi in range(len(eventsofint)):
                f = eventsofint[fi]
                if len(eventsofintsources) == 0:
                    plt.axvline(f, color=color)
                elif str(eventsofintsources[fi]) == model.source:
                    plt.axvline(f, color=color)
        else:
            for fi in range(len(eventsofint)):
                f = eventsofint[fi]
                if len(eventsofintsources) == 0:
                    ax.axvline(f, color=color)
                elif str(eventsofintsources[fi]) == model.source:
                    ax.axvline(f, color=color)

    def poltEventsWithDesc(self, score, failures, failurecodes, failuresources, model, color="red", matchcolors=[],
                           failuresplot=True,ax=None):
        if len(score) == 0:
            meanscore = 0
            stdscore = 1
        else:
            meanscore = statistics.median(score)
            stdscore = statistics.stdev(score)
            if stdscore == 0:
                stdscore = 1

        alreadyplotted = []
        positions = [-1, 0, 1]
        counter = 0
        if ax is None:
            if len(failuresources) == 0:
                for f, c in zip(failures, failurecodes):
                    findcolor = color
                    if len(matchcolors) > 0:
                        for tempdesc in matchcolors:
                            if tempdesc[0] in c:
                                findcolor = tempdesc[1]
                    plt.axvline(f, color=findcolor)
                    if c not in alreadyplotted:
                        plt.text(f, meanscore + positions[counter] * stdscore, c, color=findcolor)
                        counter += 1
                        counter = counter % len(positions)
                    if failuresplot == False:
                        alreadyplotted.append(c)
            else:
                for f, c, s in zip(failures, failurecodes, failuresources):
                    if str(s) != model.source:
                        continue
                    findcolor = color
                    if len(matchcolors) > 0:
                        for tempdesc in matchcolors:
                            if tempdesc[0] in c:
                                findcolor = tempdesc[1]
                    plt.axvline(f, color=findcolor)
                    if c not in alreadyplotted:
                        plt.text(f, meanscore + positions[counter] * stdscore, c, color=findcolor)
                        counter += 1
                        counter = counter % len(positions)
                    if failuresplot == False:
                        alreadyplotted.append(c)
            # self.getlocalContext(1*3600)
        else: # ax is not None
            if len(failuresources) == 0:
                for f, c in zip(failures, failurecodes):
                    findcolor = color
                    if len(matchcolors) > 0:
                        for tempdesc in matchcolors:
                            if tempdesc[0] in c:
                                findcolor = tempdesc[1]
                    ax.axvline(f, color=findcolor)
                    if c not in alreadyplotted:
                        ax.text(f, meanscore + positions[counter] * stdscore, c, color=findcolor)
                        counter += 1
                        counter = counter % len(positions)
                    if failuresplot == False:
                        alreadyplotted.append(c)
            else:
                for f, c, s in zip(failures, failurecodes, failuresources):
                    if str(s) != model.source:
                        continue
                    findcolor = color
                    if len(matchcolors) > 0:
                        for tempdesc in matchcolors:
                            if tempdesc[0] in c:
                                findcolor = tempdesc[1]
                    ax.axvline(f, color=findcolor)
                    if c not in alreadyplotted:
                        ax.text(f, meanscore + positions[counter] * stdscore, c, color=findcolor)
                        counter += 1
                        counter = counter % len(positions)
                    if failuresplot == False:
                        alreadyplotted.append(c)
            # self.getlocalContext(1*3600)
    def debug_forplot_score(self, listpreds, model):
        score = []
        timescore = []
        thresholds = []
        for pred in listpreds:
            if pred.description == model.name:
                score.append(float(pred.score))
                timescore.append(pred.timestamp)
                thresholds.append(pred.threshold)
        return score, timescore, thresholds

    # failures has the index, failure code the code, and failure source the source.
    # eventsofint has the time and evnets source the source.
    def evaluation(self, failures=[], failurecodes=[], failuresources=[], plotThem=False, persource=False, PH="14 days",
                   lead="1 hours", beta=2):
        dictresults = {}
        dictresults = self.evaluatemodels(self.models, dictresults, failures=failures, failurecodes=failurecodes,
                                          failuresources=failuresources, persource=persource)
        dictresults = self.evaluatemodels(self.ensembles, dictresults, failures=failures, failurecodes=failurecodes,
                                          failuresources=failuresources, persource=persource)
        Results=[]
        for keyd in dictresults.keys():
            if plotThem:
                print(f"===== {keyd} ======")
            for keyddd in dictresults[keyd].keys():
                if keyddd == "isfailure":
                    # print(f" {keyddd} : {len(dictresults[keyd][keyddd])}")
                    # print(dictresults[keyd][keyddd])
                    continue
                # print(f" {keyddd} : {len(dictresults[keyd][keyddd])}")
                # print([len(lista) for lista in dictresults[keyd][keyddd]])
            flat_trehsolds = [item for sublist in dictresults[keyd]["episodesthresholds"] for item in sublist]
            if isinstance(PH, str):
                recall, Precision, fbeta, axes = eval.myeval(dictresults[keyd]["episodescores"], flat_trehsolds,
                                                             datesofscores=dictresults[keyd]["episodesdates"], PH=PH,
                                                             lead=lead, isfailure=dictresults[keyd]["isfailure"],
                                                             plotThem=plotThem, beta=beta)
                if plotThem:
                    print(f"=======================================================")
                    print(f" RESULTS FOR {keyd}:")
                    print(f" description: {dictresults[keyd]['description']}:")
                    print(f"F{beta}: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
                    print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
                    print(f"Precission: {Precision}")
                resultdict={f"F{beta}_AD1":fbeta[0],f"F{beta}_AD2":fbeta[1],f"F{beta}_AD3":fbeta[2],
                            "AD1":recall[0],"AD2":recall[1],"AD3":recall[2],
                            "Precission":Precision}
                Results.append(resultdict)
            else:
                recall, Precision, fbeta, axes = eval.myeval_multiPH(dictresults[keyd]["episodescores"],
                                                                     dictresults[keyd]["failuretypes"], flat_trehsolds,
                                                                     datesofscores=dictresults[keyd]["episodesdates"],
                                                                     PH=PH,
                                                                     lead=lead,
                                                                     isfailure=dictresults[keyd]["isfailure"],
                                                                     plotThem=plotThem, beta=beta)
                if plotThem:
                    print(f"=======================================================")
                    print(f" RESULTS FOR {keyd}:")
                    print(f" description: {dictresults[keyd]['description']}:")
                    print(f"F{beta}: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
                    print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
                    print(f"Precission: {Precision}")
                resultdict = {f"F{beta}_AD1": fbeta[0], f"F{beta}_AD2": fbeta[1], f"F{beta}_AD3": fbeta[2],
                              "AD1": recall[0], "AD2": recall[1], "AD3": recall[2],
                              "Precission": Precision}
                Results.append(resultdict)
            if plotThem:
                plt.show()

        return Results

    def evaluatemodels(self, models, dictresults, failures=[], failurecodes=[], failuresources=[], persource=False):
        for model in models:
            dictkey = f"{model.name}_{model.uid}"
            if persource:
                dictkey = f"{model.name}_{model.uid}_{model.source}"
            sourcefailures = [(dtf, fcode) for dtf, fcode, sourfail in zip(failures, failurecodes, failuresources) if
                              str(sourfail) == str(model.source)]
            # sourceevents=[(dtf,fcode) for dtf,fcode,sourfail in zip(eventsofint,eventcodes,eventsofintsources) if sourfail==model.source]
            failuredates = [tup[0] for tup in sourcefailures]
            failurecodestemp = [tup[1] for tup in sourcefailures]
            # print(f"{model.source} filuredates: {len(failuredates)}")
            # print(f"Model {model.name} source {model.source}")
            score = []
            timescore = []
            thresholds = []
            score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            # timescore, score
            isfailure, episodescores, episodesdates, episodesthresholds, failuretypes = evalutils.brealIntoEpisodesWithCodes(
                score,
                failuredates, failurecodestemp,
                thresholds,
                timescore)
            if dictkey not in dictresults.keys():
                dictresults[dictkey] = {"isfailure": isfailure, "episodescores": episodescores,
                                        "episodesdates": episodesdates, "episodesthresholds": episodesthresholds,
                                        "description": str(model), "failuretypes": failuretypes}
            else:
                dictresults[dictkey]["isfailure"].extend(isfailure)
                dictresults[dictkey]["episodescores"].extend(episodescores)
                dictresults[dictkey]["episodesdates"].extend(episodesdates)
                dictresults[dictkey]["episodesthresholds"].extend(episodesthresholds)
                dictresults[dictkey]["failuretypes"].extend(failuretypes)

            if isinstance(model, modeltemp):
                for i in range(len(model.aggregations)):
                    dictkeyaggr = f"{dictkey}_{i}"
                    score, timescore, thresholds = self.debug_forplot_score(model.aggregatedPredictions[i], model)
                    isfailure, episodescores, episodesdates, episodesthresholds, failuretypes = evalutils.brealIntoEpisodesWithCodes(
                        score,
                        failuredates, failurecodestemp,
                        thresholds,
                        timescore)
                    if dictkeyaggr not in dictresults.keys():
                        dictresults[dictkeyaggr] = {"isfailure": isfailure, "episodescores": episodescores,
                                                    "episodesdates": episodesdates,
                                                    "episodesthresholds": episodesthresholds,
                                                    "description": f"{str(model)} aggragator:{type(model.aggregations[i])}",
                                                    "failuretypes": failuretypes}
                    else:
                        dictresults[dictkeyaggr]["isfailure"].extend(isfailure)
                        dictresults[dictkeyaggr]["episodescores"].extend(episodescores)
                        dictresults[dictkeyaggr]["episodesdates"].extend(episodesdates)
                        dictresults[dictkeyaggr]["episodesthresholds"].extend(episodesthresholds)
                        dictresults[dictkeyaggr]["failuretypes"].extend(failuretypes)
        return dictresults

    def plot_of_models_for_flask(self, failures=[], failurecodes=[], eventsofint=[], failuresources=[], eventsofintsources=[],
                   eventcodes=[], matcColor=[]):
        fig = plotly.subplots.make_subplots(rows=len(self.models), cols=1,subplot_titles=[f"{model.name} source: {model.source}" for model in self.models])

        counter=1
        for model in self.models:
            #print(f"Model {model.name} source {model.source}")

            score = []
            timescore = []
            thresholds = []
            if model.isbatch:
                score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            else:
                score, timescore, thresholds = self.debug_forplot_score(model.predictions, model)
            #plt.subplot(subnumberplot + counter)
            #plt.title(f"{model.name} source: {model.source}")
            #plt.scatter(timescore, score)

            trace = go.Scatter(x=timescore, y=score, mode='lines+markers',line=dict(color="#255b82"),name="score",legendgroup=str(counter))
            fig.add_trace(trace, row=counter, col=1)
            trace = go.Scatter(x=timescore, y=thresholds, mode='lines+markers',line=dict(color="#822b25"),name="threshold",legendgroup=str(counter))
            fig.add_trace(trace, row=counter, col=1)

            if len(timescore)>0:
                begintime=timescore[0]
                endtime=timescore[-1]


                if len(eventcodes) == 0:
                    color="magenta"
                    for fi in range(len(eventsofint)):
                        f = eventsofint[fi]
                        if f > begintime and f < endtime:
                            fig.add_vline(x=f, line_color=color, row=counter, col=1)
                else:
                    color = "magenta"
                    for fi in range(len(eventsofint)):
                        f = eventsofint[fi]
                        if f > begintime and f < endtime:
                            code = eventcodes[fi]
                            fig.add_vline(x=f,text=code, line_color=color, row=counter, col=1)

                # FAILURES
                if len(failurecodes) == 0:
                    color = "red"
                    for fi in range(len(failures)):
                        f = failures[fi]
                        if f > begintime and f < endtime:
                            fig.add_vline(x=f, line_color=color, row=counter, col=1)
                else:
                    # WITH DESC
                    color = "red"
                    for fi in range(len(failures)):
                        f = failures[fi]
                        code = failurecodes[fi]
                        if f > begintime and f < endtime:
                            fig.add_vline(x=f, text=code, line_color=color, row=counter, col=1)




            counter += 1
        fig.update_layout(
            autosize=False,
            height=200+100 * len(self.models),
            legend_tracegroupgap=54+200/len(self.models)
        )
        return fig.to_json()

