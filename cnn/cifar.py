#!/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import pickle


def main():
    pass


def load_data(filename: str) -> dict:
    with open(filename, 'rb') as file:
        read_dict = pickle.load(file)
    return read_dict


if __name__ == '__main__':
    main()
