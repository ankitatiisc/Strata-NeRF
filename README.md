# Strata-NeRF: Neural Radiance Fields for Stratified Scenes

This repository contains the code release for :
[Strata-NeRF](https://ankitatiisc.github.io/Strata-NeRF/),

This implementation is written in [JAX](https://github.com/google/jax), and
is a fork of [mip-NeRF360](https://github.com/google-research/multinerf).
This is research code, and should be treated accordingly.

The link to download our dataset can be found in our [project page](https://ankitatiisc.github.io/Strata-NeRF/) 

## Setup

```
# Clone the repo.
git clone https://github.com/ankitatiisc/Strata-NeRF.git
cd Strata-NeRF

# Make a conda environment.
conda create --name strata_nerf python=3.9
conda activate strata_nerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Running

Example scripts for training, evaluating, and rendering can be found in
`scripts/`. You'll need to change the paths to point to wherever the datasets
are located. [Gin](https://github.com/google/gin-config) configuration files
for our model and some ablations can be found in `configs/`.
Please set appropriate near and far place values in the config file depending on the number of levels in the scene.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.
