#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


def main():
    # The MLSP 2014 Schizophrenia training set is very small, below the 100
    # row limit required by DataRobot. We'll upsample it to make it run.
    upsample_schizo_train()


def upsample_schizo_train():
    df_schizo_train = pd.read_csv('MLbench/Schizophrenia/Schizophrenia_train.csv')

    # Since we'll upsample the rows, it would be wiser to use a different partitioning
    # strategy. We'll first assign a unique ID to each row, then duplicate the rows
    # and use the (now duplicated) row ID as a group partitioning feature in DataRobot.
    df_schizo_train['rowid'] = range(len(df_schizo_train))

    # Repeat every row twice.
    df_schizo_train = pd.concat([df_schizo_train, df_schizo_train])

    df_schizo_train.to_csv(
        'MLbench/Schizophrenia/Schizophrenia_train_upsampled.csv',
        header=True,
        index=None,
        float_format='%.6f',
    )


if __name__ == '__main__':
    main()
