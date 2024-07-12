import unittest


if __name__ == '__main__':

    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner()

    suite = loader.discover(start_dir='tests')
    runner.run(suite)
