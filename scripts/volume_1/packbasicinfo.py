import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt
import functools

from getPath import *
pardir = getparentdir()
common_path = pardir+'/scripts/common'

import sys
sys.path.append(common_path)
from commonLib import *

from plot_sametime import *

interval = 6

def cmp_datetime(a, b):
    tempa = datetime.strptime(a[0],"%H:%M:%S")
    tempb = datetime.strptime(b[0],"%H:%M:%S")
    if tempa>tempb: 
        return -1
    elif tempa<tempb:
        return 1
    else:
        return 0
    
def getbasicinfo():
    resdic,holidaydic = get_totaldata()
    ids = ['1-0','1-1','2-0','3-0','3-1']
    for id in ids:
        times = resdic[id]
        times = sorted(times.items(),key = functools.cmp_to_key(cmp_datetime))
        # print(times)
        i = 0
        l = len(times)
        for i in range(l):
            
        for time in times:
            print(time[0])
            plt.title(str(id)+" "+time[0])
            plt.plot(resdic[id][time[0]])
            if i%interval==0 and not i==0:
                plt.show()
            i+=1
                
    
if __name__ == "__main__":
    getbasicinfo()