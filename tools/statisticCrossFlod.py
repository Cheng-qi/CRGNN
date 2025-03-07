import pandas as pd
import re
import numpy as np
if __name__=="__main__":
    savepath = "best_results/iemocap/results/DialogueGCN"
    nums = np.array([1085,1023,1151,1031,1241],int)
    statistic = []
    for fold in [1,2,3,4,5]:
        logPath = f"{savepath}/fold{fold}/mylog.txt"
        with open(logPath, "r") as f:
            contents = f.read()
        res = re.findall("test_wacc=0.(\d{4}), test_uacc=0.(\d{4}), test_single_acc=N:[0|1].(\d{4}),A:[0|1].(\d{4}),S:[0|1].(\d{4}),H:[0|1].(\d{4})", contents)
        statistic.append(list(map(lambda x: float(x)/100, res[0])))
    statistic = np.array(statistic, float)
    weight_mean_acc = np.around(((statistic * nums.reshape(5,-1)).sum(0) / nums.sum()), 2)
    mean_acc = np.around(statistic.mean(0), 2)
    # print("weight_mean_acc")
    # print("WA   |   UA  |   N  |   A   |   S   |   H  ")
    # print(" | ".join((map(str, list(weight_mean_acc)))))
    print("weight_mean_acc")
    print("N  |   A   |   S   |   H  |   WA   |   UA  |   ")
    print(" | ".join((map(str, list(weight_mean_acc)[2:]+list(weight_mean_acc)[:2]))))
    # print("mean_acc")
    # print("WA   |   UA  |   N  |   A   |   S   |   H  ")
    # print(" | ".join((map(str, list(mean_acc)))))


    

    
    

