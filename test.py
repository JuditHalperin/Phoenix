import unittest, warnings


if __name__ == '__main__':

    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner()

    suite = loader.discover(start_dir='tests')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')
        warnings.filterwarnings('ignore', category=UserWarning, message='Features .* are constant')

        runner.run(suite)
