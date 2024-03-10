# AutoEval reproducibility

This repository contains the code to reproduce the results of "AutoEval Done Right: Using Synthetic Data for Model Evaluation"

For this purpose, clone the repository.
Next, download ImageNet, following instructions from [here](http://image-net.org/download-images) and place it in the `data` folder. 
The folder should contain the following files:
- `data/imagenet/raw_data/ILSVRC2012_devkit_t12.tar.gz`
- `data/imagenet/raw_data/ILSVRC2012_img_val.tar`

Then install snakemake (instructions [here](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)) and run the following command to reproduce the results:

```bash
snakemake --use-conda --cores 1
```