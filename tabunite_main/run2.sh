# # bank cardio churn

# python main.py --dataname bank --method i2bflow --mode train --gpu 0
# python main.py --dataname cardio --method i2bflow --mode train --gpu 0
# python main.py --dataname churn_modelling --method i2bflow --mode train --gpu 1
# python main.py --dataname bank --method dilflow --mode train --gpu 1
# python main.py --dataname cardio --method dilflow --mode train --gpu 2
# python main.py --dataname churn_modelling --method dilflow --mode train --gpu 2
# python main.py --dataname cardio --method dilflow --mode train --gpu 2
# python main.py --dataname churn_modelling --method dilflow --mode train --gpu 2
# python main.py --dataname cardio --method dilflow --mode sample --gpu 2
# python main.py --dataname churn_modelling --method dilflow --mode sample --gpu 2

# -------------------------------------------------------------------------------------------------------------------
NUM_TRAINS=1
SAMPLES_PER_TRAIN=20
declare -a arr=("stroke")
for i in "${arr[@]}"
do
    for k in $(seq 1 $NUM_TRAINS)
    do
        python main.py --dataname "$i" --method vae --mode train --gpu 2
        python main.py --dataname "$i" --method tabsyn --mode train --gpu 2
        for j in $(seq 1 $SAMPLES_PER_TRAIN)
        do
            python main.py --dataname "$i" --method tabsyn --mode sample --save_path "synthetic_multi/tabsyn_${i}_${k}-${j}.csv"
            python eval/eval_mle.py --dataname "$i" --model tabsyn --path "synthetic_multi/tabsyn_${i}_${k}-${j}.csv"
        done
    done
done