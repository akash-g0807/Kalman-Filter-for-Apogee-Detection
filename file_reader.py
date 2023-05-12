import csv
from collections import defaultdict
import numpy as np

columns = defaultdict(list)

with open('2021-09-05 14 49 Small Mike.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k


columns['time'] = columns['time'][227:]
columns['altitude'] = columns['altitude'][227:]
columns['velocity'] = columns['velocity'][227:]
columns['acc_x'] = columns['acc_x'][227:]
columns['acc_y'] = columns['acc_y'][227:]
columns['acc_z'] = columns['acc_Z'][227:]
columns['acc_tot'] = columns['acc_tot'][227:]


columns['time'] = columns['time'][227:]
columns['altitude'] = columns['altitude'][227:]
columns['velocity'] = columns['velocity'][227:]
columns['acc_x'] = columns['acc_x'][227:]
columns['acc_y'] = columns['acc_y'][227:]
columns['acc_tot'] = columns['acc_tot'][227:]

for i in range(0, len(columns['time'])):
    columns['time'][i] = float(columns['time'][i])
    columns['altitude'][i] = float(columns['altitude'][i])
    columns['velocity'][i] = float(columns['velocity'][i])
    columns['acc_x'][i] = float(columns['acc_x'][i])
    columns['acc_y'][i] = float(columns['acc_y'][i])
    columns['acc_tot'][i] = float(columns['acc_tot'][i])

def variance(data):
   # Number of observations
     n = len(data)
    # Mean of the data
     mean = sum(data) / n
     # Square deviations
     deviations = [(x - mean) ** 2 for x in data]
     # Variance
     variance = sum(deviations) / n
     return variance

print(columns["altitude"])

print(variance(columns['altitude']))