
#Used for code performance measurement
def calculateTimeDifference(startTime, endTime):
    #Find out the time difference in seconds
    timeDifferenceInSeconds = (endTime - startTime)
    
    #Translate the difference in minutes and seconds
    minutes = round(timeDifferenceInSeconds / 60)
    seconds = timeDifferenceInSeconds % 60
    
    #Create string to print
    stringToPrint = (str(minutes) + " minutes and " + str(seconds) + " seconds")
    return(stringToPrint)

#Calculate the time difference between 2 dates in seconds
def calculateTimeDif(datetimeA, datetimeB):   
    if datetimeA is None or datetimeB is None:
        return None
    # Get the difference between datetimes (as timedelta)
    dateTimeDelta = datetimeA - datetimeB

    # Find Delta in seconds
    dateTimeDeltaInSeconds = dateTimeDelta.total_seconds()
    
    return(dateTimeDeltaInSeconds)