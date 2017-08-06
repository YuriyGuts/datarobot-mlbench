#!/usr/bin/env python3

import pandas as pd


def main():
    # The MLSP 2014 Schizophrenia training set is very small, below the 100
    # row limit required by DataRobot. We'll upsample it to make it run at least.
    upsample_schizo_train()


def upsample_schizo_train():
    df_schizo_train = pd.read_csv('MLbench/Schizophrenia/Schizophrenia_train.csv')
    df_schizo_train = pd.concat([df_schizo_train, df_schizo_train])
    df_schizo_train.to_csv(
        'MLbench/Schizophrenia/Schizophrenia_train_upsampled.csv',
        header=True,
        index=None,
        float_format='%.6f',
    )


if __name__ == '__main__':
    main()
