import unittest, warnings
from scripts.utils import show_runtime


@show_runtime
def test_all():
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner()

    suite = loader.discover(start_dir='tests')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')
        warnings.filterwarnings('ignore', category=UserWarning, message='Features .* are constant')

        runner.run(suite)


if __name__ == '__main__':
    test_all()
