from aggregate_date import *
from createmodel import *
from aggregate_predict_data import *
from predict import *

def integrate_main(isVal):
    aggregate_main(0)
    creatmodel_main()
    if isVal:
       aggregate_main(isVal)
    else:      
        aggregate_predict_main(isVal)
    predict_main(isVal)
    
if __name__=="__main__":
    integrate_main(1)
    