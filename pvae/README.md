# Penalized Variational Autoencoder for Molecular Design
Official implementation of our paper in molecular design https://chemrxiv.org/articles/Penalized_Variational_Autoencoder_for_Molecular_Design/7977131/2

# Requirements
- Ubuntu
- python (version >= 3.6)
- pytorch (version >= 1.1)
- RDkit (version >=  2017.09)

# Notes:
- In this repository we only include the first part of our implementation, and joint training with the property will be released in the future. 
- For the training we used the ZINC dataset

# Training:
- The pre-trained model on ZINC dataset is available at model/zinc_pre_trained.pytorch
- if you are willing to train the model from scratch train_pure_smiles.py should be executed.
- To change parameters, please visit param.py


# Molecule generation (latent space sampling)
- In order to gnerate new molecules, we feed a SMILES structure encode them to the latent vector, then we perturbate the latent vector and decode them.
- To execute: molecule_generation.py or molecule_generation.ipynb

# Contact
For any inquiry contact sadegh.mohammadi@bayer.com

