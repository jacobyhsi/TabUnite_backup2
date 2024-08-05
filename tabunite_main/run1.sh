# bank cardio churn

# python main.py --dataname bank --method vae --mode train --gpu 1
# python main.py --dataname cardio --method vae --mode train --gpu 1
# python main.py --dataname churn_modelling --method vae --mode train --gpu 1

# python main.py --dataname bank --method tabsyn --mode train --gpu 1
# python main.py --dataname cardio --method tabsyn --mode train --gpu 1
# python main.py --dataname churn_modelling --method tabsyn --mode train --gpu 1

# python main.py --dataname bank --method tabsyn --mode sample
# python main.py --dataname cardio --method tabsyn --mode sample
# python main.py --dataname churn_modelling --method tabsyn --mode sample

# -------------------------------------------------------------------------------------------------------------------
NUM_TRAINS=1
SAMPLES_PER_TRAIN=20
declare -a arr=("stroke")
for i in "${arr[@]}"
do
    for k in $(seq 1 $NUM_TRAINS)
    do
        python main.py --dataname "$i" --method dicflow --mode train --gpu 1
        for j in $(seq 1 $SAMPLES_PER_TRAIN)
        do
            python main.py --dataname "$i" --method dicflow --mode sample --save_path "synthetic_multi/dicflow_${i}_${k}-${j}.csv"
            python eval/eval_mle.py --dataname "$i" --model dicflow --path "synthetic_multi/dicflow_${i}_${k}-${j}.csv"
        done
    done
done