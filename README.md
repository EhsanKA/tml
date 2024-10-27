## Introduction

<!-- DropQ -->

TML is a Python package for dropout-based uncertainty quantification in binary classification tasks for pruning the uncertain data. It’s designed for PyTorch Lightning users who need to perform high-confidence predictions, dataset pruning, and robust model training. TML accepts custom architectures, enabling flexible and reliable predictions in real-world scenarios.


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
python run_pipeline.py --config tml/configs/config.yaml
```

### Python

Please check [notebooks](notebooks/comparing_outputs_tf_torch.ipynb) for comparing the existing tf version (DigiPico) and the new torch implementation.


## Results for test1:

## Results

| PyTorch | TensorFlow |
|---------|------------|
| ![Germline](notebooks/torch_Germline.png) | ![Germline](notebooks/tf_Germline.png) |
| ![Probability Score](notebooks/torch_Probability_Score.png) | ![Probability Score](notebooks/tf_Probability_Score.png) |
| ![ROC](notebooks/torch_ROC.png) | ![ROC](notebooks/tf_ROC.png) |
| ![UTDs](notebooks/torch_UTDs.png) | ![UTDs](notebooks/tf_UTDs.png) |


## Dataset

Complete Datasets are available on ??

## Model Checkpoints 
See [example notebook]() for usage

## Disclaimer

All data and model checkpoints for tml are licensed under CC BY 4.0, permitting non-commercial use.