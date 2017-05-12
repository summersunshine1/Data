from aggreagate_discretization import *
from createDiscretemodel import *
from aggregate_discrete_predict import *
from predict_discrete import *

def integrate_main(isval):
    aggregate_main(0)
    select_arr = create_path_model_main()
    if isval:
        aggregate_main(1)
    else:
        aggregate_predict_path_main(0)
    predict_path_main(isval, select_arr)
    
if __name__ == "__main__":
    integrate_main(1)