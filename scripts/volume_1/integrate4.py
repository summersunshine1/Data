from predict_by_new_factor import *
from create_neighbour_model import *
from pack_new_volume_factor import *

def integrate_main():
    aggregate_main(0)
    minsize = create_model()
    aggregate_main(1)
    predict_by_new_factor_main([])
    
if __name__=="__main__":
    integrate_main()