import numpy as np
from numpy import genfromtxt

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
################################ FILE READING #############################################
columns = defaultdict(list)

with open('ValkH399.txt') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
                                 
for i in range(0, len(columns['time'])):
    columns['time'][i] = float(columns['time'][i])
    columns['altitude'][i] = float(columns['altitude'][i])
    #columns['velocity'][i] = float(columns['velocity'][i])
    columns['acc_x'][i] = float(columns['acc_x'][i])
    columns['acc_y'][i] = float(columns['acc_y'][i])
    columns['acc_z'][i] = float(columns['acc_z'][i])
    #columns['acc_tot'][i] = float(columns['acc_tot'][i])




x_0 = columns['altitude'][0]  # replace values for x_0, v_0  with the values from the sensor
#v_0 = columns['velocity'][0]    # replace values for x_0, v_0  with the values from the sensor
a = -9.81

delta_t = 0.0052

f = open("output.csv", "w")
f2 = open("original.csv", "w")

################################ FILE READING #############################################

################################ KALMAN FILTER #############################################

# Mapping matrix
H = np.array([[1, 0, 0], [0, 0, 1]])
H_transpose = np.transpose(H)
C = np.array([[1, 0, 0], [0, 0, 1]])

# Measurement noise covariance
R = np.array([[40, 0.00], [0.00, 0.012]]) 

# Process noise covariance
Q = np.array([[0, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

# Mwasurement noise 
Z = np.array([[0], [0]])            # change the last array to the measured values uncertaintiyu from the sensor

# Transition state matrix
A = np.matrix([[1,delta_t,(delta_t*delta_t)/2], [0, 1, delta_t], [0, 0, 1]])
A_transpose = np.transpose(A)

# Initial state
def initial_state(x_0, a):
    X_k = np.array([[x_0], [a*delta_t], [a]])
    return X_k

# Predicted state
def predict_state(X_k1):
    X_kp = np.dot(A, X_k1)
    return X_kp

# Predicted covariance
def predict_covariance(P_k1):
    P_kp = np.add(np.dot(np.dot(A, P_k1), A_transpose), Q)
    return P_kp

# Kalman gain
def kalman_gain_value(P_kp):
    K = np.dot(np.dot(P_kp, H_transpose), np.linalg.inv(np.add(np.dot(np.dot(H, P_kp), H_transpose), R)))
    return K

# Adjusted measured values
def measured_matrix(Y_km): 
    Y_k = Y_km  + Z 
    return Y_k

# Updated state
def update_state(X_kp, K, Y_k):
    X_k = np.add(X_kp, np.dot(K, np.subtract(Y_k, np.dot(H, X_kp))))
    return X_k

# Updated covariance
def update_covariance(K, P_kp):
    P = np.dot(np.subtract(np.identity(3), np.dot(K, H)), P_kp)
    return P

# Kalman filter
def kalman_filter():
    X_k = initial_state(x_0,a)
    P_k = np.array([[1, 0.00, 0.00], [0.00, 1, 0.00], [0.00, 0.00, 1]]) # initial covariance matrix

    for i in range(1, len(columns['time'])):
        X_k1 = X_k
        P_k1 = P_k
        Y_km = np.array([[columns['altitude'][i]],[columns['acc_z'][i]]])

        X_kp = predict_state(X_k1)
        P_kp = predict_covariance(P_k1)
        K = kalman_gain_value(P_kp)
        Y_k = measured_matrix(Y_km)
        X_k = update_state(X_kp, K, Y_k)
        P_k = update_covariance(K, P_kp) 

        print(columns['time'][i],X_k.item(0))
        f.write(str(columns['time'][i]) + "," + str(X_k.item(0))+"\n")       # Kalman filter output
        f2.write(str(columns['time'][i]) + "," + str(columns['altitude'][i])+"\n")  # Original data


kalman_filter()

################################ KALMAN FILTER #############################################