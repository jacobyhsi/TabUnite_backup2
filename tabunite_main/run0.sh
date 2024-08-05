# bank cardio churn

# python main.py --dataname bank --method i2bflow --mode train --gpu 0
# python main.py --dataname syn1 --method pskflow --mode train --gpu 0
# python main.py --dataname bank --method i2bflow --mode train --gpu 0
# python main.py --dataname cardio --method i2bflow --mode train --gpu 0
# python main.py --dataname churn_modelling --method i2bflow --mode train --gpu 0

# python main.py --dataname bank --method i2bflow --mode sample
# python main.py --dataname cardio --method i2bflow --mode sample
# python main.py --dataname churn_modelling --method i2bflow --mode sample

# python eval/eval_mle.py --dataname bank --model i2bflow --path synthetic/bank/i2bflow.csv
# python eval/eval_mle.py --dataname cardio --model i2bflow --path synthetic/cardio/i2bflow.csv
# python eval/eval_mle.py --dataname churn_modelling --model i2bflow --path synthetic/churn_modelling/i2bflow.csv

# python eval/eval_mle.py --dataname bank --model dicflow --path synthetic/bank/dicflow.csv
# python eval/eval_mle.py --dataname cardio --model dicflow --path synthetic/cardio/dicflow.csv
# python eval/eval_mle.py --dataname churn_modelling --model dicflow --path synthetic/churn_modelling/dicflow.csv

# python eval/eval_mle.py --dataname bank --model tabsyn --path synthetic/bank/tabsyn.csv
# python eval/eval_mle.py --dataname cardio --model tabsyn --path synthetic/cardio/tabsyn.csv
# python eval/eval_mle.py --dataname churn_modelling --model tabsyn --path synthetic/churn_modelling/tabsyn.csv

# python eval/eval_mle.py --dataname bank --model tabddpm --path synthetic/bank/tabddpm.csv
# python eval/eval_mle.py --dataname cardio --model tabddpm --path synthetic/cardio/tabddpm.csv
# python eval/eval_mle.py --dataname churn_modelling --model tabddpm --path synthetic/churn_modelling/tabddpm.csv

# python main.py --dataname syn1 --method pskflow --mode train --gpu 0
# python main.py --dataname syn1 --method pskflow --mode sample


# -------------------------------------------------------------------------------------------------------------------
NUM_TRAINS=1
SAMPLES_PER_TRAIN=20
declare -a arr=("stroke")
for i in "${arr[@]}"
do
    for k in $(seq 1 $NUM_TRAINS)
    do
        python main.py --dataname "$i" --method i2bflow --mode train --gpu 0
        for j in $(seq 1 $SAMPLES_PER_TRAIN)
        do
            python main.py --dataname "$i" --method i2bflow --mode sample --save_path "synthetic_multi/i2bflow_${i}_${k}-${j}.csv"
            python eval/eval_mle.py --dataname "$i" --model i2bflow --path "synthetic_multi/i2bflow_${i}_${k}-${j}.csv"
        done
    done
done