from options import Options

from datasets import CIFAR10
from datasets import MNIST
from utils import set_random_seed
from train_helper.mem_mnist_trainer import MEM_Mnist_Trainer

def train_mnist(opt):
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # Build dataset and model
    dataset = MNIST(path='/MNIST')

    # Set up result helper and perform test
    helper = MEM_Mnist_Trainer(opt, dataset)
    helper.train()

def main():

    # Parse command line arguments
    args = Options().parse()

    # Lock seeds
    set_random_seed(30101990)

    # Run test
    if args.model == 'mnist':
        train_mnist(opt=args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


# Entry point
if __name__ == '__main__':
    main()
