#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import pickle
import os

DATA_DIR = './cifar-10-batches-py/'
SUMMARY_DIR = '/tmp/tf_summary/'


def main():
    prepare()


def load_data(filename: str) -> dict:
    with open(filename, 'rb') as file:
        read_dict = pickle.load(file, encoding='iso-8859-1')
    return read_dict


def read_data() -> np.ndarray:
    pass


def prepare():
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)


if __name__ == '__main__':
    main()
