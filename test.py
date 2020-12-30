import argparse
from argparse import Namespace

from datasets import CIFAR10
from datasets import MNIST
from models import MEMCIFAR10
from models import MEMMNIST
import torch
import torch.nn as nn
from result_helpers import MEMResultHelper
from utils import set_random_seed

def test_mnist(args):
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # Build dataset and model
    dataset = MNIST(path=args.path)
    model = MEMMNIST(input_shape=dataset.shape, code_length=64, cpd_channels=100, mem_dim=100, shrink_thres=0.5/100).cuda().eval()

    # Set up result helper and perform test
    helper = MEMResultHelper(dataset, model, checkpoints_dir=args.checkpoints, output_file='mem_mnist.txt')
    helper.test_one_class_classification()


def test_cifar(args):
    # type: () -> None
    """
    Performs One-class classification tests on CIFAR
    """

    # Build dataset and model
    dataset = CIFAR10(path=args.path)
    model = MEMCIFAR10(input_shape=dataset.shape, code_length=64, cpd_channels=100, mem_dim=500, shrink_thres=1.0 / 500).cuda().eval()

    # Set up result helper and perform test
    helper = MEMResultHelper(dataset, model, checkpoints_dir=args.checkpoints, output_file='mem_cifar.txt')
    helper.test_one_class_classification()

def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                             'Choose among `mnist`, `cifar10`', metavar='')
    parser.add_argument('path', type=str,
                        help='The file path of the dataset to perform tests', metavar='')
    parser.add_argument('checkpoints', type=str,
                        help='The checkpoints path of the dataset to perform tests', metavar='')

    return parser.parse_args()

def main():

    # Parse command line arguments
    args = parse_arguments()

    # Run test
    if args.dataset == 'mnist':
        test_mnist(args=args)
    elif args.dataset == 'cifar10':
        test_cifar(args=args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')