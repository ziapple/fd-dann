source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
from test import test

if __name__ == '__main__':
    test(source_dataset_name, 5)
    test(target_dataset_name, 5)
