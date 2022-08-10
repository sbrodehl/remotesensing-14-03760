# End-to-End Prediction of Lightning Events from Geostationary Satellite Images
[![Python 3.9](https://img.shields.io/badge/python-3.9-brightgreen.svg)](#deep-satellite-based-detection-and-forecast-of-convective-cells)
[![Linting Python](https://img.shields.io/github/workflow/status/sbrodehl/remotesensing-14-03760/Linting%20Python?label=linting)](https://github.com/sbrodehl/remotesensing-14-03760/actions?query=workflow%3A%22Linting+Python%22)
[![DOI 10.3390/rs14153760](https://img.shields.io/badge/DOI-10.3390%2Frs14153760-brightgreen)](https://doi.org/10.3390/rs14153760)

This is the official repo for the paper ["End-to-End Prediction of Lightning Events from Geostationary Satellite Images"](https://doi.org/10.3390/rs14153760).

## Abstract

> While thunderstorms can pose severe risks to property and life, forecasting remains challenging, even at short lead times, as these often arise in meta-stable atmospheric conditions.
> In this paper, we examine the question of how well we could perform short-term (up to 180 min) forecasts using exclusively multi-spectral satellite images and past lighting events as data.
> We employ representation learning based on deep convolutional neural networks in an "end-to-end" fashion.
> Here, a crucial problem is handling the imbalance of the positive and negative classes appropriately in order to be able to obtain predictive results (which is not addressed by many previous machine-learning-based approaches).
> The resulting network outperforms previous methods based on physically based features and optical flow methods (similar to operational prediction models) and generalizes across different years.
> A closer examination of the classifier performance over time and under masking of input data indicates that the learned model actually draws most information from structures in the visible spectrum, with infrared imaging sustaining some classification performance during the night.
>
> _from ["End-to-End Prediction of Lightning Events from Geostationary Satellite Images"](https://doi.org/10.3390/rs14153760)_.

## Getting Started

### Prerequisites

DFCC is based on the following libraries and their versions:

* [Python 3.9](https://www.python.org/downloads)
* [PyTorch 1.8.2 LTS](https://pytorch.org/get-started)

Additionally, the Python packages as listed in the [requirements.txt](requirements.txt) file must be installed.  
We recommend to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

### Usage

#### Package Structure

The [`src`](src) directory contains all the code to run the experiments.
Here, everything is split up into different modules:
- [`data`](src/data) includes everything related to data, from different data sources to actual data loading
- [`events`](src/events) includes the learning range rate test (LRRT) and code to run experiments
- [`experiments`](src/experiment) includes different experiments
- [`log`](src/log) includes logging
- [`model`](src/model) includes model code
- [`optimizer`](src/optimizer) includes everything related to optimizer, lr scheduling, loss and loss weighting
- [`skill`](src/skill) includes skill scores, such as CSI, FAR and POD
- [`system`](src/system) includes system settings
- [`utils`](src/utils) includes utilities used by other modules

The [`scripts`](scripts) directory contains scripts to create figures displayed in our publication.

#### Experiments

##### Settings

| Lead Time | Bottleneck Channels | Weight Decay | Regularization Factor |
|----------:|--------------------:|-------------:|----------------------:|
|     0 min |                  16 |    5 x 10^-6 |                   0.1 |
|    30 min |                  81 |    7 x 10^-6 |                  0.15 |
|    60 min |                  97 |    7 x 10^-6 |                   0.2 |
|    90 min |                 106 |    7 x 10^-6 |                   0.2 |
|   120 min |                 114 |    1 x 10^-5 |                   0.3 |
|   180 min |                 124 |    1 x 10^-5 |                   0.3 |


##### Figure 4

Run the following code with the settings above to get a trained model for the given lead time:
```python
python run.py experiment
  --SEVIRI.roots.train "${train-data}"
  --SEVIRI.roots.test "${test-data}"
  --SEVIRI.roots.val "${validation-data}"
  --LINET.db_path "${linet.db}"
  --LINET.lead_time "${lead-time}"
  --regularizer_factor "${reg-factor}"
  --sgdw.weight_decay "${weight-decay}"
  --channel_hierarchy_depth "${bottleneck-channels}"
```

The testing step will dump detailed model statistics which can be used with [figure_4.py](scripts/figure_4.py).

##### Figure 6

For Figure 6 train models with varying input channels, e.g. VIS+PER, WV+PER and IR+PER, and pass the testing details to [figure_6.py](scripts/figure_6.py).  
For the sake of simplicity, we use the option `--zero_out` to replace inputs of the given channel with zeros during training.

```python
python scripts/figure_6.py
  -lt 120
  --vis-per vis+per-testing_model_details.json
  --wv-per wv+per-testing_model_details.json
  --ir-per ir+per-testing_model_details.json
  --json complete-testing_model_details.json
  --persistence testing_persistence_details.json
```

## Citation

If you find the code useful for your research, please cite (see [CITATION](CITATION) for details):

- Brodehl, S.; Müller, R.; Schömer, E.; Spichtinger, P.; Wand, M. End-to-End Prediction of Lightning Events from Geostationary Satellite Images. Remote Sens. 2022, 14, 3760. https://doi.org/10.3390/rs14153760

## Versioning

We use [SemVer](http://semver.org/) for versioning.
For the versions available, see the [releases](https://github.com/sbrodehl/DFCC/releases) and [tags](https://github.com/sbrodehl/DFCC/tags) on this repository.

## Authors
* **Sebastian Brodehl** / [@sbrodehl](https://github.com/sbrodehl)

## License
With the exceptions noted below, this project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

- [lr_finder.py](src/events/LRRT/lr_finder.py) taken from [github.com/davidtvs/pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) (MIT license) and adapted to be used in this project.
- [modelsummary](src/log/modelsummary/__init__.py) taken from [github.com/sksq96/pytorch-summary](https://github.com/sksq96/pytorch-summary) (MIT license) and adapted to be used in this project.
