# TabUnite: An Efficient Encoding Framework for Tabular Data Generation
Official Implementation of the Paper - TabUnite: An Efficient Encoding Framework for Tabular Data Generation.

## Usage

tabunite_main: baseline experiments

tabunite_census-synthetic: census synthetic dataset

tabunite_syn_qual: qualitative synthetic dataset

tabunite_syn_quant: quantitative synthetic dataset

## Environment

Clone this repository and navigate to it in your terminal.

Create environment. This environment can be used for everything apart from eval_quality:

```
conda create -n tabunite python=3.10
conda activate tabunite
```

Install pytorch via conda:

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

and other dependencies:
```
pip install -r requirements.txt

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

Any other missing dependencies can be installed using pip. Once all the dependencies are installed, the scripts should run accordingly.

For eval_quality, create the following environment:
```
conda create -n tabunite_quality python=3.10
conda activate tabunite_quality

pip install synthcity
pip install category_encoders
pip install tomli
pip install tomli_w
pip install zero
pip install scipy
pip install click==8.1.7
pip install scipy==1.12.0
```



## Datasets

Baseline datasets can be downloaded within the tabunite_main directory as follows:

```
python download_dataset.py
python process_dataset.py
```

For the "Census Synthetic" dataset, download the census.csv dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/116/us+census+data+1990) where you change the .data file to .csv instead. Save it within the repository then replace the path within tabunite_census-synthetic/dataset.py. The dataname for this dataset is denoted as: syn1.

## Training, Sampling and Evaluation

Training:
```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```
```
python main.py --dataname beijing --method i2bflow --mode train
```

Sampling:
```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode sample
```
```
python main.py --dataname beijing --method i2bflow --mode sample
```

Evaluation:
```
python eval/eval_mle.py --dataname [NAME_OF_DATASET] --model [NAME_OF_BASELINE_METHODS] --path [PATH_TO_SYNTHETIC_DATA]
python eval/eval_density_cde.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
python eval/eval_density_pwc.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
python eval/eval_quality.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
python eval/eval_detection.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```
```
python eval/eval_mle.py --dataname beijing --model i2bflow --path synthetic/beijing/i2bflow.csv
```
