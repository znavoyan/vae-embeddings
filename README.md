## Improving VAE based molecular representations for compound property prediction
Here we present the implementation of our paper [Improving VAE based molecular representations for compound property prediction](https://arxiv.org/abs/2201.04929)

In this paper we propose a simple method to improve chemical property prediction performance of machine learning models by incorporating additional information on correlated molecular descriptors in the representations learned by variational autoencoders.

## Requirements
Please see the `requirements.txt`

## Variational autoencoders

In this work we have used two types of Variational autoencoders: 
- **Chemical VAE** (or as shortly refered in this work: **CVAE**) - proposed by Gómez-Bombarelli et al. [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://doi.org/10.1021/ACSCENTSCI.7B00572]), [github](https://github.com/aspuru-guzik-group/chemical_vae). Only slight changes are made in the original code of CVAE.
- **Penalized VAE** (or as shortly refered in this work: **PVAE**) - proposed by S. Mohammadi et al. [ Penalized Variational Autoencoder for Molecular Design](https://doi.org/10.26434/CHEMRXIV.7977131.V2), [github](https://github.com/InvincibleIRMan/pvae). As the codes do not contain the implementation of joint training with the property, we have implementated the functionality by ourselfs.

## Installation
For running the codes, you need to install a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with all the required packages by running the following codes. Note that, for running CVAE training, you have to use `environment_cvae.yml` file and for PVAE and other trainings of downstream tasks `environment.yml` file.
```
git clone https://github.com/znavoyan/vae-fingerprints.git
cd vae-fingerprints
conda env create -f environment.yml
```

## Training
For each specific downstream task, for example solubility prediction (LogS), the training process consists of three steps:
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
with `-d` we specify the model's directory, in which you must have `exp.json` file, containing all the parameters for training the model.


**PVAE**
The training of PVAE is similar to CVAE, only in this case the training of variational autoencoder with or without property prediction is done with separate scripts: `train_pure_smiles.py` for VAE without property predictor and `train_prop.py` for VAE with property predictor. For both cases with `-d` we specify the model's directory, in which you must have `params.json` file, containing all the parameters for training the model.
```
cd pvae
python train_prop.py -d ./models/zinc_logp_196/
```

### Step 2: Extracting molecular embeddings
In this step, having the pre-trained VAE model, we can encode the molecules from downstream task's dataset to high dimensional embeddings. The code below shows an example of getting embeddings for Solubility prediction dataset using PVAE trained with MolLogP property predictor:
```
python src/fingerprints/pvae.py --input ../data/logS/processed/final_logS_6789.csv --model_dir ./pvae/models/zinc_logp_196/ --output ../data/logS/processed_with_pvae/final_logS_pvae_logp_196.csv
```
With `--input` we specify the path to downstream task's dataset, `--model_dir` shows the path to variational autoencoder trained during Step 1, and with `--output` we show where the new dataset with embeddings will be saved.

### Step 3: Training of model for downstream task

```
python src/train.py --property logS --data ./data/logS/processed_with_pvae/final_logS_pvae_logp_196_6668.csv --save_dir ./models/cv10_logS_6668_pvae_emb_logp_196 --feature vae_emb --fold_indices_dir ./data/logS/fold_indices_pvae/
```

## License
Apache License Version 2.0