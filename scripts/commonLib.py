import numpy as np
from datetime import datetime,timedelta
import math
from sklearn.preprocessing import OneHotEncoder
from commonLib import *

def getphase(interval,date):
    a = math.ceil(interval/9)
    b = interval%9
    phase = a*3
    if b<=4 and b>0:
        phase = phase-3
    if phase == 24:
        trace_time = datetime.strptime(date, "%Y/%m/%d")
        year = trace_time.year
        month = trace_time.month
        day = trace_time.day
        # day = day+1
        # if day>days[month]:
            # month += 1
            # day = 1
        date = str(year)+"/"+str(month)+"/"+str(day)
        phase = 0
    return phase,date

def isholiday(date):
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    year = trace_time.year
    month = trace_time.month
    day = trace_time.day
    if (trace_time.month==10 and trace_time.day>=1 and trace_time.day<=7) or (trace_time.month==9 and trace_time.day==30):
        return 1
    return 0
    
def zeroNormalize(arr):
    mu = np.average(arr)
    sigma = np.std(arr)
    if sigma == 0:
        return arr
    return (arr-mu)/sigma
    
def linearNormalize(arr):
    max = np.amax(arr)
    min = np.amin(arr)
    if max == min:
        return arr
    return (arr-min)/(max-min)
            
def getweekday(date):
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    weekday = trace_time.strftime("%w")
    return weekday

def isfixedWeekDay(date, fixed):
    weekday = int(getweekday(date))
    if fixed == weekday:
        return 1
    return 0
        
def getNormalizeWeekday(dates):
    l = len(dates)
    weekdays = []
    for i in range(l):
        weekday = getweekday(dates[i])
        weekdays.append(int(weekday))
    weekdays = np.array(weekdays)
    weekdays = zeroNormalize(weekdays)
    weekdic = {}
    for i in range(l):
        weekdic[dates[i]] = weekdays[i]
    return weekdic

def getNormalizeHoliday(dates):
    l = len(dates)
    holidays = []
    for i in range(l):
        holiday = isholiday(dates[i])
        holidays.append(holiday)
    holidays = zeroNormalize(holidays)
    holidaydic = {}
    for i in range(l):
        holidaydic[dates[i]] = holidays[i]
    return holidaydic
    
def getweekarr(dates):
    l = len(dates)
    weekdays = []
    for i in range(l):
        weekday = getweekday(dates[i])
        temp = []
        temp.append(int(weekday))
        weekdays.append(temp)
    weekdays = np.array(weekdays)
    return weekdays
    
def getholidayarr(dates):
    l = len(dates)
    holidays = []
    for i in range(l):
        holiday = isholiday(dates[i])
        temp = []
        temp.append(holiday)
        holidays.append(temp)
    holidays = np.array(holidays)
    return holidays
  
def encoder(arr):
    ohe = OneHotEncoder(sparse=False)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr)   


  