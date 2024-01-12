import os
import pandas as pd

daymin= pd.to_datetime("2022-03-26 06:25:19")
daystart= pd.to_datetime("2023-01-01 00:00:00")

diff = daystart-daymin




# Replace 'path/to/csv_folder' with the actual path to your folder containing CSV files
csv_folder_path = '/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/NavarchosPdM/tempData/NavarchosData/delta'

# Create an empty list to store DataFrames
dfs = []

# Iterate through the CSV files in the folder
for filename in os.listdir(csv_folder_path):
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(csv_folder_path, filename)

    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path,index_col=0,header=0)
        df.index=pd.to_datetime(df.index)
        fdate=pd.to_datetime("2023-05-23 00:00:00")+diff
        df = df.loc[:fdate]
        df.to_csv(csv_file_path)



