import time

import pandas as pd

from utils.structure import Eventpoint
from metafussion.fussionSystem import fuss
import setmodels
def PHPrepare(DefinedPHs,failurecodes,ph,lead):
    definedcodes = [tup[0] for tup in DefinedPHs]
    remainingun = [codedesc for codedesc in failurecodes if codedesc not in definedcodes]
    remainingun.append("non failure")
    remainingun = list(set(remainingun))
    PHtemp = [(codedesk, ph, lead) for codedesk in remainingun]
    DefinedPHs.extend(PHtemp)
    PH = [(tup[0], tup[1]) for tup in DefinedPHs]
    lead = [(tup[0], tup[2]) for tup in DefinedPHs]
    return PH,lead
def notinlist(listoroi, desc):
    for oros in listoroi:
        if oros in desc:
            return False
    return True



def formulateData(dfs, events, tdcevents, sources=[], excluded=["Accident", "Tyre"], reseteventsParameter=["Standard service", "Oil change"]):


    unicodes = [f"{ev['action_type']}_{ev['desc']}" for i, ev in events.iterrows() if
                (ev["action_type"] == "R" and notinlist(excluded, ev["desc"])) or ev["desc"] in reseteventsParameter]
    unicodes = list(set(unicodes))
    alltimes = []
    alltypesofdata = []
    resetscodeslis = []
    for dff, sourccc in zip(dfs, sources):
        alltimes.extend([i for i in dff.index])
        alltypesofdata.extend([f"data:{sourccc}" for i in dff.index])
        resetscodeslis.append([(code, sourccc) for code in unicodes])
        dff.index = [dt.tz_localize(None) for dt in dff.index]

    events['dt'] = [dt.tz_localize(None) for dt in events['dt']]
    alltimes.extend([i for i in events["dt"]])
    alltypesofdata.extend(["maintenance" for i in events.index])

    tdcevents['dt'] = [dt.tz_localize(None) for dt in tdcevents['dt']]
    alltimes.extend([i for i in tdcevents["dt"]])
    alltypesofdata.extend(["dtc" for i in tdcevents.index])


    alltimes = [dt.tz_localize(None) for dt in alltimes]

    time_type = list(set(zip(alltimes, alltypesofdata)))

    timee = [x for x, _ in sorted(time_type)]
    typee = [y for _, y in sorted(time_type)]

    return timee, typee, resetscodeslis, events, tdcevents, dfs





def check_if_events_exist(datadf,vehicle,unicodes):
    df = pd.read_csv("tempData/NavarchosData/newerservices.csv")
    df['dt'] = pd.to_datetime(df['dt'])#, format='mixed')

    vehicle_related = df[df['vehicle_id'] == int(vehicle)]
    begin = min(datadf.index)
    end = max(datadf.index)
    tupss = [(des, time) for des, time in zip(vehicle_related['desc'], vehicle_related['dt']) if
             time >= begin and (f"R_{des}" in unicodes or f"S_{des}" in unicodes)]
    tupss = list(set(tupss))
    return len(tupss)>0




def localNavarchosSimulation(datasetname = "n=300_slide=50",oilInReset=False,ExcludeNoInformationVehicles=False,printThemSim=False):
    if ExcludeNoInformationVehicles:
        sources = ['25', '5', '4', '16', '28', '27', '14', '20', '26', '33', '24', '18', '29', '31', '2', '30', '7', '21', '13', '17', '34', '32', '23', '11', '8', '9']
    else:
        sources = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                   '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                   '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']





    excluded = ["Accident", "Tyre", "door", "Whell","Wheel", "Rims", "Lamp", "Horn", "Strut", "Αντιστάσεις", "Windsfield",
                "DPF", "Blue", "blue", "brake", "Battery", "Brake", "Επιδι", "Steer", "Shock",
                "motor driven fan", "Spark","Engine base","Mirror"]

    excludedresets = ["Accident", "Tyre", "door", "Whell", "Wheel", "Rims", "Lamp", "Horn", "Strut", "Αντιστάσεις",
                      "Windsfield",
                      "DPF", "Blue", "blue", "brake", "Battery", "Brake", "Επιδι", "Steer", "Shock", "Spark"]

   ## NOW: RESET ON OIL CHANGE AND ACCIDENTS
    if oilInReset:
        reseteventsParameter = ["Standard service" ,"Oil change"]
    else:
        reseteventsParameter = ["Standard service"]


    events = pd.read_csv("tempData/NavarchosData/newerservices.csv", index_col=0)
    events['dt'] = pd.to_datetime(events['dt'])#, format='mixed')
    unicodes = [f"{ev['action_type']}_{ev['desc']}" for i, ev in events.iterrows() if
                (ev["action_type"] == "R" and notinlist(excluded, ev["desc"])) or ev["desc"] in reseteventsParameter]
    unicodes = list(set(unicodes))
    dfs = []

    sourceori = []
    for vehicle in sources:
        datadf = pd.read_csv(f"tempData/NavarchosData/{datasetname}/{vehicle}.csv", index_col=0)


        datadf.index = pd.to_datetime(datadf.index)
        datadf = datadf[~datadf.index.duplicated(keep='first')]

        sourceori.append(vehicle)
        dfs.append(datadf)

    sources = sourceori
    print(f"Total vehcicles: {len(sources)}")
    print(sources)
    events = pd.read_csv("tempData/NavarchosData/newerservices.csv", index_col=0)
    events['dt'] = pd.to_datetime(events['dt'])#, format='mixed')
    #print(events.head())

    tdcevents = pd.read_csv("tempData/NavarchosData/dtc_all.csv", index_col=0)
    tdcevents['dt'] = pd.to_datetime(tdcevents['dt'])


    timee, typee, resetscodeslis, events, tdcevents, dfs = formulateData(dfs, events, tdcevents,
                                                                         sources=sources,
                                                                         excluded=excludedresets, reseteventsParameter=reseteventsParameter)
    failuretimes, failurecodes, failuresources, eventsofint, eventsofintsources, eventscodes = failureandEvents(events,
                                                                                                                excluded,
                                                                                                                tdcevents,
                                                                                                                sources)

    models = []
    ensembles = []
    for source, resetcode in zip(sources, resetscodeslis):
        sourcemodels, tempensebles = setmodels.modelsnavinit(resetcode, source)
        models.extend(sourcemodels)
        ensembles.extend(tempensebles)

    fussiontest = fuss(models, ensembles)

    start = time.time()
    runSimulationNavarchos(timee, typee, events, tdcevents, dfs, fussiontest, sources,printThem=printThemSim)
    end = time.time()
    print(f"execution time: {end - start}")

    fussiontest.write_results()
    PlotsDebug=True
    # testing predicting failure of temperature related only


    matcColor = [("pending", "yellow"), ("storred", "black"),("fatigueDriving","grey")]



    beta = 0.5
    plothem = PlotsDebug

    fussiontest.debug_plot(failuretimes, failurecodes, eventsofint, failuresources=failuresources,
                           eventsofintsources=eventsofintsources, eventcodes=eventscodes, matcColor=matcColor)

    ph = "15 days"
    lead = "1 hours"
    DefinedPHs = []

    PH, lead = PHPrepare(DefinedPHs, failurecodes, ph, lead)

    fussiontest.evaluation(failures=failuretimes, failurecodes=failurecodes, failuresources=failuresources,
                           plotThem=plothem, persource=False, PH=PH, lead=lead, beta=beta)


    plothem = PlotsDebug
    ph = "30 days"
    lead = "1 hours"
    DefinedPHs = []

    PH, lead = PHPrepare(DefinedPHs, failurecodes, ph, lead)

    fussiontest.evaluation(failures=failuretimes, failurecodes=failurecodes, failuresources=failuresources,
                           plotThem=plothem, persource=False, PH=PH, lead=lead, beta=beta)



    # fussiontest.debug_plot_analytical(failuretimes, failurecodes, eventsofint, failuresources=failuresources,
    #                                   eventsofintsources=eventsofintsources, eventcodes=eventscodes,
    #                                   matcColor=matcColor)
def failureandEvents(events,excluded,tdcevents,sources):

    failuretimes = [ev['dt'] for i, ev in events.iterrows() if
                    ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]
    failurecodes = [ev['desc'] for i, ev in events.iterrows() if
                    ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]
    failuresources = [ev['vehicle_id'] for i, ev in events.iterrows() if
                      ev["action_type"] == "R" and str(ev["vehicle_id"]) in sources and notinlist(excluded, ev["desc"])]

    eventsofint = [ev['dt'] for i, ev in events.iterrows() if (ev["action_type"] == "S" or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False)) and str(
        ev["vehicle_id"]) in sources]
    eventsofint.extend([ev['dt'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])

    eventsofintsources = [ev['vehicle_id'] for i, ev in events.iterrows() if (ev["action_type"] == "S" or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False)) and str(
        ev["vehicle_id"]) in sources]
    eventsofintsources.extend([ev['vehicle_id'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])

    eventscodes = [ev['desc'] for i, ev in events.iterrows() if (ev["action_type"] == "S" or (
            ev["action_type"] == "R" and notinlist(excluded, ev["desc"]) == False)) and str(
        ev["vehicle_id"]) in sources]
    eventscodes.extend([ev['dtc_status'] for i, ev in tdcevents.iterrows() if str(ev["vehicle_id"]) in sources])
    return failuretimes,failurecodes,failuresources,eventsofint,eventsofintsources,eventscodes



def runSimulationNavarchos(timee, typee, events, tdcevents, dfs, fussiontest, sources,printThem=False):
    if printThem:
        print(sources)
    for dttime, dtypee in zip(timee, typee):
        if dtypee.split(":")[0] == "data":
            tempsource = dtypee.split(":")[1]
            row = dfs[sources.index(tempsource)].loc[dttime].values
            fussiontest.collect_data(row, dttime, tempsource)
        elif dtypee == "maintenance":
            rows = events[events["dt"] == dttime]
            for q in range(len(rows.index)):
                row = rows.iloc[q]
                eventpoint = Eventpoint(f"{row['action_type']}_{row['desc']}", str(row['vehicle_id']), dttime)
                if printThem:
                    print(f"{dttime}: {row['desc']}")
                fussiontest.collect_event(eventpoint)
        elif dtypee == "dtc":
            rows = tdcevents[tdcevents["dt"] == dttime]
            for q in range(len(rows.index)):
                row = rows.iloc[q]
                eventpoint = Eventpoint(f"{row['dtc_status']}_{row['dtc_data']}", str(row['vehicle_id']), dttime)
                # print(f"{dttime}: {row['dtc_status']}_{row['dtc_data']}")
                fussiontest.collect_event(eventpoint)
