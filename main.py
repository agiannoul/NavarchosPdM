import simulateNavarchos

if __name__ == '__main__':

    #simulateNavarchos.localNavarchosSimulation(datasetname = "raw",oilInReset=False,ExcludeNoInformationVehicles=False,printThemSim=False)
    simulateNavarchos.localNavarchosSimulation(datasetname = "correlation",oilInReset=False,ExcludeNoInformationVehicles=True,printThemSim=False)
    #simulateNavarchos.localNavarchosSimulation(datasetname = "mean",oilInReset=False,ExcludeNoInformationVehicles=True,printThemSim=False)
    #simulateNavarchos.localNavarchosSimulation(datasetname = "delta",oilInReset=False,ExcludeNoInformationVehicles=False,printThemSim=False)


