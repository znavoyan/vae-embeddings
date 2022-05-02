## Improving VAE based molecular representations for compound property prediction
Here we present the implementation of our paper [Improving VAE based molecular representations for compound property prediction](https://arxiv.org/abs/2201.04929)

In this paper we propose a simple method to improve chemical property prediction performance of machine learning models by incorporating additional information on correlated molecular descriptors in the representations learned by variational autoencoders.

## Variational autoencoders

In this work we have used two types of Variational autoencoders: 
- **Chemical VAE** (or as shortly refered in this work: **CVAE**) - proposed by Gómez-Bombarelli et al. [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://doi.org/10.1021/ACSCENTSCI.7B00572]), [github](https://github.com/aspuru-guzik-group/chemical_vae). Only slight changes are made in the original code of CVAE.
- **Penalized VAE** (or as shortly refered in this work: **PVAE**) - proposed by S. Mohammadi et al. [ Penalized Variational Autoencoder for Molecular Design](https://doi.org/10.26434/CHEMRXIV.7977131.V2), [github](https://github.com/InvincibleIRMan/pvae). As the codes do not contain the implementation of joint training with the property, we have implementated the functionality by ourselves.

## Installation
For running the codes, you need to install a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with all the required packages by running the following commands. Note that, for CVAE you have to use `environment_cvae.yml` file while for the PVAE and the other trainings of downstream tasks `environment.yml` file.
```
git clone https://github.com/znavoyan/vae-fingerprints.git
cd vae-fingerprints
conda env create -f environment.yml
conda env create -f environment_cvae.yml
```

## Training
For each specific downstream task, e.g. solubility prediction (LogS), the training process consists of three steps:
1. Train variational autoencoder (CVAE or PVAE) 
2. Extract the molecular embeddings from the trained VAE
3. Train another neural network for the downstream task

### Step 1: Training of VAE
For the training of VAE we are using 250k ZINC dataset placed in `data/zinc` folder.
**CVAE**
```
cd chemical_vae
python -m chemvae.train_vae_new -d models/zinc_logp_196/
```
`-d` specifies the model's directory, which should include `exp.json` file with all the parameters for training the model.

**PVAE**
The training of PVAE is similar to CVAE, with only one exception. To train VAE with property prediction use `train_pure_smiles.py` script, while for training without property prediction use `train_prop.py` script. For both cases use `-d` to specify the model's directory, which must include `params.json` file with all the parameters for training the model.
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
After extracting molecular embeddings, we can now train the model for downstream task. The idea of ResNet is taken from the paper proposed by Cui et al [Improved Prediction of Aqueous Solubility of Novel Compounds by Going Deeper With Deep Learning](https://doi.org/10.3389/FONC.2020.00121/BIBTEX). As the paper did not provide the ResNet codes, we included the implementation of the model using the given hyperparameters. The following code shows an example of training ResNet model for Solubility (LogS) prediction task:
```
python src/train.py --property logS --data ./data/logS/processed_with_pvae/final_logS_pvae_logp_196_6668.csv --save_dir ./models/cv10_logS_6668_pvae_emb_logp_196 --feature vae_emb --fold_indices_dir ./data/logS/fold_indices_pvae/ --model ResNet
```
The meaning of the arguments:
```
--property: downstream task's name, in this work it can be 'logS', 'logBB' or 'logD'
--data: path to downstream task's dataset
--save_dir: shows where to save all the training results and models
--feature: shows which representations should be used as input to the model. Here this arguments can only be 'vae_emb' as we are focused only on molecular embeddings extracted from VAE, but for future, other representations can be used as well, such as fingerprints, descriptors or some combinations of several representations
--fold_indices_dir: directory containing indices for each fold, as we are performing cross validation. If there are no indices, i.e. the training is done for a new dataset, by specifying only the name of the directory, new separation will be done and the indices will be saved in that directory. Number of folds is determined by fold_num*repeat_folds expression.
--model: model type for downstream task's training, can be 'ResNet', 'MLP' or 'LR'
```
Other arguments not included in the command above which have default values:
```
--fold_num: number of folds for cross validation, default = 10
--repeat_folds: number of times cross validation is repeated, default = 1
--start_fold: shows which fold the training should start/continue from, in case the training is interrupted for some reasons, default = 1
--epochs: number of epochs for ResNet, if not specified, the default values for LogS, LogD and LogBB are 2000, 1500 and 85 respectively
--learning_rate: learning rate for ResNet 
--batch_size: batch size for ResNet 
--l2_wd: L2 weight decay regularization for ResNet 
--mlp_max_iter: maximum number of iterations for MLP
```

## Evaluation
You can get the predictions for each fold and look at the metrics by running `test.py` file:
```
python src/test.py --experiment ./models/cv10_logS_6668_pvae_emb_logp_196 --model ResNet
```
`--experiment` argument specifies the directory of experiment, i.e. a folder containing all the trained model(s) and parameters for a downstream task, and the  `--model` argument shows the trained model type, from which we woluld like to get the predictions, as already said, it can be 'ResNet', 'MLP' or 'LR'. Predictions are saved under `predictions_ResNet` folder in case of `--model` being ResNet and so on.

## License
Apache License Version 2.0