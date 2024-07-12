import unittest
from scripts.utils import define_set_size


class UtilTest(unittest.TestCase):
    
    def test_set_size_definition(self):
        self.assertEqual(define_set_size(20, 0.5, 15), 15)
        self.assertEqual(define_set_size(8, 0.5, 10), 8)
        self.assertEqual(define_set_size(80, 0.25, 10), 20)
        self.assertEqual(define_set_size(80, 0.5, 10), 35)  # as 40 is not in SIZES


if __name__ == '__main__':
    unittest.main()
