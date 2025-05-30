# Round_and_Round_RoPE

## Setup

First create the conda environment: 

```shell
conda create -n "RoPE" python==3.10
conda activate RoPE 
```

Then install the required dependencies (including PySvelte and fixing a bug on displaying a table in `notebooks/successor_heads.ipynb`) by running:   

```shell
bash setup.sh
```


## Training our model

In order to use our model, you first need to *train* the _tokenizer_ and the _model_. To train the tokenizer, run the following command:
```
python train_tokenizer.py
```

Afterwards, train the model using:

```
python train_model.py
```