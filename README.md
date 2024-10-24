## Introduction
A comprehensive package for Transductive Machine Learning
??


## Installation

```bash
pip install git+https://github.com/EhsanKA/tml.git

```

From source
```bash
yes | conda create --name tml python=3.11.10
conda activate tml
pip install -e .
```

<!-- For contribution please install in dev mode and use pre-commit
```bash
yes | conda create --name pika python=3.10
conda activate pika
pip install -e ".[dev]"
pre-commit install
``` -->

## Usage

### Commandline


Nested keys can be parsed as below as long as all keys are present in the config
```bash
python run_pipeline.py --config configs/config.yaml
```

### Python

Please check [notebooks](?) for examples.


## Dataset

Complete Datasets are available on ??

## Model Checkpoints 
See [example notebook]() for usage

## Disclaimer

All data and model checkpoints for tml are licensed under CC BY 4.0, permitting non-commercial use.