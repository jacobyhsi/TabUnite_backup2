import json
import numpy as np

datasets = ["bank", "cardio", "stroke"]
methods = ["i2bflow", "dicflow", "tabsyn", "tabddpm"]
train_runs = 1
sample_runs = 20
# methods = ["dilflow", "oheflow", "i2bflow", "tabflow",  "tabddpmi2b", "tabddpmohe", "tabddpmdic", "tabddpm"]

print("Train runs: ", train_runs)
print("Sample runs: ", sample_runs)

for d in datasets:
    for m in methods:
        cur_list = []
        for train in range(train_runs):
            for sample in range(sample_runs):
                try:
                    f = open(f'eval/mle/{d}_multi/{m}_{d}_{train+1}-{sample+1}.json') # Path to your eval/mle json file
                    # print(f'eval/mle/{d}_multi/{m}_{d}_{train+1}-{sample+1}.json')
                    data = json.load(f)
                    if (d == "beijing" or d == "news" or d=="census"):
                        cur_res = data['best_rmse_scores']['XGBRegressor']['RMSE']
                        # print(cur_res)
                    else:
                        cur_res = data['best_auroc_scores']['XGBClassifier']['roc_auc']
                    cur_list.append(cur_res)
                except Exception as e:
                    print(e)
                    pass
        arr = np.array(cur_list)
        print(f'{d}_{m}')
        print(f'mean: {np.round(np.mean(arr),3)}')
        print(f'std: {np.round(np.std(arr),3)}')
        print()
        