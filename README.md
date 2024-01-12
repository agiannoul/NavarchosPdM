# Implementation of "" paper.

## Data

Folder temp data contain the raw vehicular under "raw" folder. Data are splitted to different files, each one associated with a unique identifier (correspoding to different vehicles).

Moreover there availiable logs from maintenances considering Services and Repairs in the **newerservices.csv** file. Where the data, type and abastrac description is provided, along with the corresponding vehicle id.

For a small fraction of vehicles there available the produced DTC codes in the file **dtc_all.csv**

## Running the code

Using anaconda:
```
conda create --name navpdm python=3.8
conda activate navpdm
conda install --file requirements.txt
python3 main.py
```

Non-conda enviroment:
```
# On Windows: .\navpdm\Scripts\activate
# On macOS/Linux: source navpdm/bin/activate
python -m venv navpdm
pip install -r requirements.txt
```


Along with the proposed framework the code provide a system to simulate the data in operation of the fleet, as it would be in real-life. 
The documentation regarfing fundumental methods and concepts regarding the system can be seen in docs.md

To run the similation the localNavarchosSimulation dunction can be used:

 - **datasename**: specify the aggregation of data (raw,correlation,delta,mean)
 - **oilReset**: If you want to consider oil changes as a reseting event (Look in the paper regarding the meaning of reset event)
 - **ExcludeNoInformationVehicles**: if you want to consider or not the vehicles where no data for maintenances and sevises are availiable (True to exclude them)
 - **printThemSim**: To print the results of detection and evaluation.


```python
import simulateNavarchos

if __name__ == '__main__':
  navarchos.localNavarchosSimulation(datasetname = "correlation",oilInReset=False,ExcludeNoInformationVehicles=True,printThemSim

```
To change the method used for detecting the failure use the **setmodels.py**.

There are four availiable technques: TranAD, Pair Detection, Xgboost Pair Detection and grand Inductive. 

## Illustration of Analytical Results:

![Alt Text](/images/vehicleCorrelationResults.png)


## Others

To this point, the system is build to support implementation of different techniqeus that work for streaming simulation data (i.e. data arrived one by one). The methods to work should respect the existing interface (look at existing solutions and docs.md).

