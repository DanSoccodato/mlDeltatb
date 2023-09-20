import unittest
loader = unittest.TestLoader()
start_dir = './'
suite = loader.discover(start_dir, pattern='*Test.py')

runner = unittest.TextTestRunner(verbosity=3)
runner.run(suite)
