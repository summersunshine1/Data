from handleResidual import *
from aggregate_volume_predict_data import *
from create_volume_model import *
from packThreefactor import *

ids = [1,2,3]
directions = [0,1]



def integrate_main(isvalid):
    index = 0
    for id in ids:
        for direction in directions:
            if id==2 and direction==1:
                continue
            # if id==1 and direction==1:
            print(str(id)+"-"+str(direction))
            handle_main(id, direction)
            trend_cols,residual_cols = create_main()
            aggregate_main(id, direction,isvalid)
            packThreefactor_main(id,direction,trend_cols)
            create_three_factor_model()
            index+=1
            # if index==2:
                # break
        # if index==2:
                # break 
            
if __name__ == '__main__':
    integrate_main(0) 