## Introduction
A comprehensive package for Transductive Machine Learning
??


## Setting Up the Environment

To recreate the Conda environment used for this project, please follow these steps:

1. Make sure you have Conda installed. If not, you can download and install Miniconda or Anaconda from:
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - [Anaconda](https://www.anaconda.com/products/distribution)

2. Clone this repository:

   ```sh
   git clone https://github.com/EhsanKA/tml.git
   cd tml

3. Create the Conda environment using the provided YAML file:
    
    ```sh
    conda env create --file=environment.yaml

4. Activate the environment:
    ```sh
    conda activate tml

5. install the package:

    ```sh
    pip install -e .



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