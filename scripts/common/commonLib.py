import numpy as np
from datetime import datetime,timedelta
import math
from sklearn.preprocessing import OneHotEncoder

def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth) 

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
    
def getlinks():
    links = []
    for i in range(100,124):
        links.append(i)
    return links
    
def get_link_seperate_path():
    linkseq = [[110,123,107,108],[120,117], [119,114,118], [105,100],[111,103],[122],[116,101,121,106,113],[115,102,109,104,112]]
    return linkseq
    
def get_date_num_from_timewindow(time_windows):#2016-07-19 00:00:00
    length = len(time_windows)
    if length == 1:       
        trace_starttime = datetime.strptime(time_windows[0], "%Y/%m/%d %H:%M:%S")
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        num = (trace_starttime.hour*60+trace_starttime.minute)/20
        if num==0:
            num = 72
        return date,num
    dates = []
    nums = []
    for i in range(length):
        trace_starttime = time_windows[i]
        # trace_starttime = datetime.strptime(time_windows[i], "%Y/%m/%d %H:%M:%S")
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        num = (trace_starttime.hour*60+trace_starttime.minute)/20
        if num==0:
            num = 72
        dates.append(date)
        nums.append(num)
    return dates,nums
    
def getnormtime(intervals):
    dic = {}
    norm_intervals = zeroNormalize(intervals)
    # norm_intervals = (intervals - 0)/(72)
    l = len(intervals)
    for i in range(l):
        dic[intervals[i]] = norm_intervals[i]
    return dic
    
def get_time_from_interval(date,interval):
    hour = int(interval*20/60)
    minute = int(interval*20%60)
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    year = trace_time.year
    month = trace_time.month
    day = trace_time.day
    start_time_window = datetime(year, month, day, hour, minute, 0)
    end_time_window = start_time_window + timedelta(minutes=20)
    return start_time_window,end_time_window
    
def get_intersection_toll():
    return ['A-2','A-3','B-3','B-1','C-3','C-1']
    
def get_path():
    dic = {}
    dic['A-2'] = [0,1] #[110,123,107,108][120,117]
    dic['A-3'] = [0,2,5]
    dic['B-3'] = [3,4,5]
    dic['B-1'] = [3,4,6]
    dic['C-3'] = [7,4,5]
    dic['C-1'] = [7,4,6]
    return dic

    

    
    




  