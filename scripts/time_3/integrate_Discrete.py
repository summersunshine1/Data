from aggreagate_discretization import *
from createDiscretemodel import *
from aggregate_discrete_predict import *
from predict_discrete import *

def integrate_main(isval):
    # aggregate_main(0)
    # select_arr = create_path_model_main()
    if isval:
        aggregate_main(1)
    else:
        aggregate_predict_path_main(0)
    # print(select_arr)
    arr=[]
    for i in range(9):
        temp=[]
        for j in range(129):
            temp.append(str(j))
        arr.append(temp)
    predict_path_main(isval, arr)
    
if __name__ == "__main__":
    integrate_main(0)