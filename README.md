## Improving VAE based molecular representations for compound property prediction
Here we present the implementation of our paper [Improving VAE based molecular representations for compound property prediction] https://arxiv.org/abs/2201.04929

In this paper we propose a simple method to improve chemical property prediction performance of machine learning models by incorporating additional information on correlated molecular descriptors in the representations learned by variational autoencoders

## Requirements

- Python (vesion >= 3.7)
- Tensorflow (vesion = 1.14)
- Rdkit (vesion = 2020.09.1)


## Variational autoencoders

In this work we have used two types of Variational autoencoders: 
- **Chemical VAE** (or as shortly refered in this work: **CVAE**) - proposed by Gómez-Bombarelli et al. [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://doi.org/10.1021/ACSCENTSCI.7B00572]), [github](https://github.com/aspuru-guzik-group/chemical_vae). Only slight changes are made in the original code of CVAE in this work.
- **Penalized VAE** (or as shortly refered in this work: **PVAE**) - proposed by S. Mohammadi et al. [ Penalized Variational Autoencoder for Molecular Design](https://doi.org/10.26434/CHEMRXIV.7977131.V2), [github](https://github.com/InvincibleIRMan/pvae). As mentioned in their github repository, the codes do not contain the implementation of joint training with the property, so we have done the implementation ourselfs in this repositoory


## Installation
For succesfully running the codes, you need to have an environment with all the requrired packages. For that, you can create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) virtual environment by using the `environment.yml` file in this repository by running the following commands in the terminal
```
git clone https://github.com/znavoyan/vae-fingerprints.git
cd vae-fingerprints
conda env create -f environment.yml
```
or create your own environment and install all the packages mentioned in `requirements.txt`

## Training
For each specific task (downstream task), for example solubility prediction (LogS), the training process consists of three steps:
1. Train variational autoencoder (CVAE or PVAE) 
2. Extract the molecular embeddings from the trained VAE
3. Train another neural network for downstream task

### Step 1: Training of VAE
For the training of VAE we are using 250k ZINC dataset placed in `data/zinc` folder.
**CVAE**
```
cd chemical_vae
python -m chemvae.train_vae_new -d models/zinc_logp_196/
```
with `-d` we specify the model's directory, in which you must have `exp.json` file, containing all the parameters for training the model. You can find an example of `exp.json` file below, in which the parameters are set for training the model with MolLogP property predictor:

```
{
  "name": "zinc",
  "MAX_LEN": 120,
  "train_data_file": "../data/zinc/zinc_250k_train.csv",
  "test_data_file": "../data/zinc/zinc_250k_test.csv",
  "char_file": "./models/zinc_logp_196/zinc.json",
  "encoder_weights_file": "./models/zinc_logp_196/zinc_encoder.h5",
  "decoder_weights_file": "./models/zinc_logp_196/zinc_decoder.h5",
  "prop_pred_weights_file": "./models/zinc_logp_196/zinc_prop_pred.h5",
  "reg_prop_tasks": ["MolLogP"],
  "test_idx_file": "test_idx.npy",
  "history_file": "history.csv",
  "checkpoint_path": "./",
  "do_prop_pred": true,
  "TRAIN_MODEL": true,
  "ENC_DEC_TEST": false,
  "PADDING": "right",
  "RAND_SEED": 42,
  "epochs": 120,
  "vae_annealer_start": 29,
  "dropout_rate_mid": 0.082832929704794792,
  "anneal_sigmod_slope": 0.51066543057913916,
  "recurrent_dim": 488,
  "hidden_dim": 196,
  "tgru_dropout": 0.19617749608323892,
  "hg_growth_factor": 1.2281884874932403,
  "middle_layer": 1,
  "prop_hidden_dim": 67,
  "batch_size": 182,
  "prop_pred_depth": 3,
  "lr": 0.00045619868229310396,
  "prop_pred_dropout": 0.15694573998898703,
  "prop_growth_factor": 0.99028340731314179,
  "momentum": 0.99027641036225744
}
```

**PVAE**
The training of PVAE is similar to CVAE, only in this case the training of variational autoencoder with or without property prediction is done with separate scripts: `train_pure_smiles.py` for VAE without property predictor and `train_prop.py` for VAE with property predictor. For both cases with `-d` we specify the model's directory, in which you must have `params.json` file, containing all the parameters for training the model.
```
cd pvae
python train_prop.py -d ./models/zinc_logp_196/
```
Example of `params.json` for training a model with MolLogP property predictor:

```
{
    "data_dir": "../data/zinc/",
    "data_list": ["zinc_250k_train.csv", "zinc_250k_test.csv"],
    "gpu_exist": true,
    "device_id": 0,
    "batch_size": 30,
    "hidden_size": 1024,
    "embedding_size": 30,
    "bidirectional": false,
    "epochs": 72,
    "nr_classes": 4,
    "predict_prop": true,
    "nr_prop": 1,
    "properties": ["MolLogP"],
    "rnn_type": "gru", 
    "learning_rate": 0.001,
    "latent_size": 196,
    "n_layers": 1,
    "save_every": 10,
    "word_dropout": 0.1,
    "vocab_size": 33,
    "anneal_function": "logistic",
    "k0": 2500,
    "x0": 0.0025,
    "save_dir": ""
}
```


### Step 2: Extracting molecular embeddings

In this step, having the pre-trained VAE model, we can encode the molecules from downstream task's dataset to high dimensional embeddings. The code below shows an example of getting embeddings for Solubility prediction dataset using PVAE trained with MolLogP property predictor:
```
python src/fingerprints/pvae.py --input ../data/logS/processed/final_logS_6789.csv --model_dir ./pvae/models/zinc_logp_196/ --output ../data/logS/processed_with_pvae/final_logS_pvae_logp_196.csv
```
With `--input` we specify the path to downstream task's dataset, `--model_dir` shows the path to variational autoencoder trained during Step 1, and with `--output` we show where the new dataset with embeddings will be saved.


### Step 3: Training of model for downstream task
blablabla

## License
Apache License Version 2.0