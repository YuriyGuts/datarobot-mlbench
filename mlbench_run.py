#!/usr/bin/env python3

"""
Runs the datasets from the mlbench benchmark [https://arxiv.org/abs/1707.09562v2] in DataRobot,
using the DataRobot API client. Creates multiple submission files for each dataset.
"""

import logging
import sys
import yaml

import datarobot as dr
import pandas as pd

from datetime import timedelta
from datarobot.models.modeljob import wait_for_async_model_creation


# Maximum number of DataRobot workers to use for the projects in the benchmark.
MAX_DATAROBOT_WORKERS = 4

# Maximum wait timeout for upload, modeling, and prediction jobs.
MAX_WAIT_SECONDS = int(timedelta(hours=1).total_seconds())

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s | %(levelname)8s | %(message)s')
logger = logging.getLogger(__name__)


def run_mlbench_project(mlbench_project):
    """
    Run the modeling experiment for a single dataset from mlbench.
    Get predictions from DataRobot, save them as separate submission files.
    """

    # Create a DataRobot project and run the autopilot.
    dr_project = run_dr_project(mlbench_project)

    # Retrain the best models from the leaderboard on 100% sample size.
    best_autopilot_models = get_best_autopilot_models(dr_project)
    retrained_models = retrain_models_100pct(dr_project, best_autopilot_models)
    submission_models = best_autopilot_models + retrained_models

    # Request predictions from the best autopilot models and the best retrained models.
    submissions = create_submissions(mlbench_project, dr_project, submission_models)

    # Create submission files.
    submission_filenames = [
        mlbench_project['name'] + '-dr-best-solo-model-autopilot.csv',
        mlbench_project['name'] + '-dr-best-blender-autopilot.csv',
        mlbench_project['name'] + '-dr-best-solo-model-100pct.csv',
        mlbench_project['name'] + '-dr-best-blender-100pct.csv',
    ]
    for submission_df, submission_filename in zip(submissions, submission_filenames):
        submission_df.to_csv(submission_filename, index=None, header=True)


def run_dr_project(mlbench_project):
    """
    Given the metadata for an MLbench dataset, run a modeling project in DataRobot.
    """

    logger.info('DataRobot: Creating project...')
    dr_project = dr.Project.create(
        mlbench_project['train_dataset'],
        project_name=mlbench_project['name']
    )

    logger.info('DataRobot: Aim...')
    dr_project.set_target(
        target=mlbench_project['target_name'],
        metric=mlbench_project['metric'],
        partitioning_method=dr.StratifiedCV(holdout_pct=20, reps=5),
        advanced_options=dr.AdvancedOptions(accuracy_optimized_mb=True),
        worker_count=MAX_DATAROBOT_WORKERS,
    )

    logger.info('DataRobot: Waiting for autopilot...')
    dr_project.wait_for_autopilot()

    return dr_project


def create_submissions(mlbench_project, dr_project, models):
    """
    Given the list of DataRobot models, create a submission dataframe for each model.
    """

    # Request predictions from all specified models.
    logger.info('DataRobot: Uploading the prediction dataset...')
    test_dataset = dr_project.upload_dataset(
        mlbench_project['test_dataset'],
        max_wait=MAX_WAIT_SECONDS,
        read_timeout=MAX_WAIT_SECONDS,
    )

    logger.info('DataRobot: Requesting predictions...')
    predict_jobs = [model.request_predictions(test_dataset.id) for model in models]
    for job in predict_jobs:
        job.wait_for_completion(max_wait=MAX_WAIT_SECONDS)

    # Download predictions.
    logger.info('DataRobot: Downloading predictions...')
    all_prediction_sets = [
        dr.PredictJob.get_predictions(project_id=dr_project.id, predict_job_id=job.id)
        for job in predict_jobs
    ]

    # Convert predictions to competition format.
    logger.info('Creating submissions...')
    submissions = [
        convert_predictions(mlbench_project, df_predictions)
        for df_predictions in all_prediction_sets
    ]
    return submissions


def convert_predictions(mlbench_project, df_predictions):
    """
    Convert the downloaded predictions from DataRobot to a format accepted by the competition.
    """

    df_prediction_ids = pd.read_csv(mlbench_project['test_prediction_id_dataset'])
    df_submission = pd.concat([df_prediction_ids, df_predictions['positive_probability']], axis=1)
    df_submission.rename(
        columns={'positive_probability': mlbench_project['test_prediction_column_name']},
        inplace=True,
    )
    return df_submission


def get_best_autopilot_models(dr_project):
    """
    Retrieve the best performing models (according to CV score) from the DataRobot leaderboard.
    """

    logger.info('DataRobot: Retrieving leaderboard...')
    metric = dr_project.metric
    is_cv_run = len([model for model in dr_project.get_models() if model.metrics[metric]['crossValidation']])
    evaluation_set = 'crossValidation' if is_cv_run else 'validation'

    leaderboard = sorted(
        [model for model in dr_project.get_models() if model.metrics[metric][evaluation_set]],
        key=lambda m: m.metrics[metric][evaluation_set],
        reverse=metric in ['AUC', 'Gini Norm'],
    )
    best_solo_model = next(model for model in leaderboard if model.model_category != 'blend')
    best_blender = next(model for model in leaderboard if model.model_category == 'blend')

    return [best_solo_model, best_blender]


def retrain_models_100pct(dr_project, models):
    """
    Retrain the specified list of DataRobot models on 100% sample size.
    """

    logger.info('DataRobot: Unlocking holdout...')
    dr_project.unlock_holdout()

    logger.info('DataRobot: Retraining the models...')
    retrain_job_ids = [model.train(sample_pct=100) for model in models]
    retrained_models = [
        wait_for_async_model_creation(
            project_id=dr_project.id,
            model_job_id=job_id,
            max_wait=MAX_WAIT_SECONDS,
        )
        for job_id in retrain_job_ids
    ]

    return retrained_models


def main():
    with open('mlbench.yaml') as cfg:
        projects = yaml.load(cfg)

    for project in projects:
        try:
            logger.info('Running project: %s', project['name'])
            run_mlbench_project(project)
            logger.info('Project %s completed successfully', project['name'])
        except Exception:
            logger.error('Project %s failed', project['name'], exc_info=True)
            pass


if __name__ == '__main__':
    main()
