import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', PendingDeprecationWarning)

import unittest
from scripts.utils import show_runtime


@show_runtime
def test_all():
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner()

    suite = loader.discover(start_dir='tests')
    runner.run(suite)


if __name__ == '__main__':
    test_all()
