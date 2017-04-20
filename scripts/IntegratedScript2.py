from volume_predict import *
from handleResidual import *
from aggregate_volume_predict_data import *
from create_volume_model import *

ids = [1,2,3]
directions = [0,1]

def integrate_main():
    for id in ids:
        for direction in directions:
            if id==2 and direction==1:
                continue
            # handle_main(id, direction)
            # create_main()
            aggregate_main(id, direction)
            predict_main(id, direction)
            
if __name__ == '__main__':
    integrate_main()