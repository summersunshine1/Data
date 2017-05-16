import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt

from getPath import *
pardir = getparentdir()
common_path = pardir+'/scripts/common'

import sys
sys.path.append(common_path)
from commonLib import *

from plot_sametime import *

def cmp_datetime(a, b):
    tempa = datetime.strptime(a,"%H:%M:%S")
    tempb = datetime.strptime(b,"%H:%M:%S")
    if tempa>tempb: 
        return -1
    elif tempa<tempb:
        return 1
    else:
        return 0
    
def getbasicinfo():
    resdic,holidaydic = getvolumeinfo()
    ids = ['1-0','1-1','2-0','3-0','3-1']
    for id in ids:
        times = resdic[id]
        times = sorted(times.items(),cmp = cmp_datetime, key = lambda d:d[0])
        # print(times)
        i = 0
        for time in times:
            print(time[0])
            plt.title(str(id)+" "+time[0])
            plt.plot(resdic[id][time[0]])
            if i%6==0 and not i==0:
                plt.show()
            i+=1
                
    
if __name__ == "__main__":
    getbasicinfo()