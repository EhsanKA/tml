## Introduction
A comprehensive package for Transductive Machine Learning
??


## Installation

<!-- the environment was created with this:
`conda env export --from-history > environment.yaml`
Note: consider removing the prefix line in the `environment.yaml` -->

```bash
git clone https://github.com/EhsanKA/tml.git
cd tml
conda env create --file environment.yaml
conda activate tml
pip install .
```


<!-- ## Installation

```bash
pip install git+https://github.com/EhsanKA/tml.git

``` -->


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