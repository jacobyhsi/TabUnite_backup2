import json
import numpy as np

# datasets = ["adult", "beijing", "default", "magic", "news", "shoppers"]
datasets = ["syn1"]
# num_train = 2
samples_per_train = 3
# methods = ["i2bflow", "dilflow", "oheflow", "tabflow", "i2bddpm", "dicddpm", "oheddpm", "tabddpm"]
# methods = ["i2b", "dil", "tab"]

num_train = [0]
methods = ["tabddpm"]

for d in datasets:
    for m in methods:
        cur_list = []
        for train in num_train:
            for sample in range(samples_per_train):
                try:
                    f = open(f'eval/mle/{d}_1/{m}_{d}_{train+1}-{sample+1}.json')
                    data = json.load(f)
                    if (d == "beijing" or d == "news" or d == "syn1"):
                        cur_res = data['best_rmse_scores']['XGBRegressor']['RMSE']
                    else:
                        cur_res = data['best_auroc_scores']['XGBClassifier']['roc_auc']
                    cur_list.append(cur_res)
                except Exception as e:
                    # print(e)
                    pass
        arr = np.array(cur_list)
        print(f'{d}_{m}')
        print(f'mean: {np.mean(arr):.3f}')
        print(f'std: {np.std(arr):.3f}')
        print()
        