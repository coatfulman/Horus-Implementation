import pandas as pd
import numpy as np
def check(lowY, highY, posY):
    if posY >= lowY and posY<=highY:
        diff1 = posY - lowY
        diff2 = highY - posY
        if diff1 > diff2:
            posY = highY
        else:
            posY = lowY
    return posY

def findActualLocation(startTime , endTime, stopTime, maxTime):
    #### startTime is the starting time of current time window. 
    #### endTime is the end time of current time window
    #### stopTime is a constant. Take it to be 10 seconds
    #### maxTime is maximum time observed in current movement trace i.e. basically last occuring timestamp value in current file. 
    personLocation = pd.read_table('person_location.txt', delimiter="\t")
    personLocation.columns = ['x', 'y']
    personLocation['x'] = personLocation['x'] * 0.0254 ### changing to meters
    personLocation['y'] = personLocation['y'] * 0.0254
    distance = 0
    time = maxTime
    for index, row in personLocation.iterrows():
        if index != 0:
            distance = distance + np.sqrt((row['x'] - X) * (row['x'] - X) + (row['y'] - Y) * (row['y'] - Y))
        time = time - stopTime
        X = row['x']
        Y = row['y']
    movingSpeed = distance/time
    
    locationAtTime = endTime
    time = 0
    for index, row in personLocation.iterrows():
        if index != 0:
            d = np.sqrt((row['x'] - X) * (row['x'] - X) + (row['y'] - Y) * (row['y'] - Y))
            prevTime = time
            time = time + (d/movingSpeed)
            if time >= locationAtTime:
                dirX = row['x'] - X
                dirY = row['y'] - Y
                posY = (Y + (dirY * movingSpeed * (locationAtTime - prevTime) / d))
#                 posY = check(lowY=80,highY=101,posY=posY)
#                 posY = check(lowY=140,highY=161,posY=posY)
#                 posY = check(lowY=200,highY=221,posY=posY)
                posY = check(lowY=2.032,highY=2.5654,posY=posY)
                posY = check(lowY=3.556,highY=4.0894,posY=posY)
                posY = check(lowY=5.08,highY=5.6134,posY=posY)
                return (X + (dirX * movingSpeed * (locationAtTime - prevTime) / d)) , posY
        time = time + stopTime
        if time >= locationAtTime:
            return row['x'], row['y']
        X = row['x']
        Y = row['y']
    return X,Y


###############################
# We assume a time window of 10 seconds
# time_window = 10.0
# stop_time = 10.0
# If you want to find ground truth location of 1st time window, you call it as :
# i = 0
# x,y = findActualLocation(startTime=time_window*i , endTime=time_window*(i+1), stopTime=stop_time, maxTime=maxTime)
################################
