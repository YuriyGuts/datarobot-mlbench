# datarobot-mlbench

Runs the Kaggle datasets from the [mlbench](https://arxiv.org/abs/1707.09562) benchmark [H. Zhang et al., 2017]
in [DataRobot](https://www.datarobot.com/) cloud, using the [DataRobot API client](http://pythonhosted.org/datarobot/).
Creates multiple submission files for each dataset.

## How to Use

1. [Download the mlbench dataset](https://ds3lab.org/mlbench/) and extract it into this directory.
2. [Configure your access credentials](http://pythonhosted.org/datarobot/setup/configuration.html) for the DataRobot API.
3. Run `mlbench_preprocess.py`.
4. Run `mlbench_run.py`.
5. Run `mlbench_postprocess.py`.

The script will run all datasets in DataRobot and create 4 submission files for each dataset.
These submission files can be uploaded to Kaggle for scoring.
