from predict_by_new_factor import *
from create_neighbour_model import *
from pack_new_volume_factor import *

def integrate_main():
    aggregate_main(0)
    # meanx,stdx = create_model()
    features,mean_x1,std_x1 = create_model()
    aggregate_main(1)
    predict_by_new_factor_main([],features,mean_x1,std_x1)
    
if __name__=="__main__":
    integrate_main()