<div align="center">

# ctdg

<h4>A Continuous-Time Dynamic Graphs library</h4>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

## Installation

Before installing the library, you need to install some dependencies that are not specified in the `pyproject.toml` file, so they will not be installed automatically by `pip`. In particular, you need to install `torch` and `torch-scatter`. If you need to use the HDBSCAN-based rewirers of TIPAR and CONNECT, you also need to install the `cuml` library. Note that the `cuml` library depends on `cupy` that requires a CUDA installation on your machine, thus we recommend using a conda environment to install it as it will install all the necessary dependencies for you.

```bash
conda create -n ctdg python=3.10

conda install -c rapidsai -c conda-forge -c nvidia cuml=24.08 'cuda-version>=12.0,<=12.5'
# then install torch and torch-scatter
```

Currently, we do not provide a PyPI package for this library, but you can easily install it from this repository using `pip`:

```bash
pip install git+https://github.com/FrancescoGentile/ctdg.git
```

If you instead want to modify the library, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/FrancescoGentile/ctdg.git
cd ctdg
pip install -e .
```

## Usage

Other than using the library as a dependency in your project, you can also use it a command-line tool to train and evaluate the provided models on a bunch of different tasks. At the moment, the library supports the following tasks:

- **future link prediction**: predict the presence of an interaction between two nodes at a given time in the future. The datasets that can be used for this task are the following: Wikipedia, Reddit, MOOC, LASTFM (of course, you can also use your own dataset).
- **macroscopic cascade prediction**: predict the number of nodes that will be activated by a cascade process. The datasets that can be used for this task are the following: Christianity, Android, Douban, Memetracker, Twitter.
- **microscopic cascade prediction**: predict the next node that will be activated by a cascade process. The same datasets used for the macroscopic cascade prediction task can be used for this task as well.

The models currently available in the library are the following:
- TGN
- TIGE (this is TIGER without restarts)
- TIPAR
- CONNECT

All implemented models can be trained on the tasks mentioned above. To train a model on a task, you can use the following command:

```bash
ctdg {model_name} --phase train --task {task_name} --config {config_file}
```

where `{model_name}` is the lowercase name of the model you want to train, `{task_name}` should one of "link-prediction", "cascade-prediction" (to specify which type of cascade prediction task you want to train the model on, pass a `--mode` argument with the value "macro" or "micro"), and `{config_file}` is the path to a configuration file that specifies the dataset to use, the structure of the model and other hyperparameters. As configuration files, we take inspiration from detectron2 and we use python-based configuration files. You can find some examples in the `configs` directory.