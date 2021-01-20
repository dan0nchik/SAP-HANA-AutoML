import unittest
from pipeline.input import Input
import pandas as pd


class InputTestCase(unittest.TestCase):

    def test_url(self):
        i = Input()
        df = i.load_from_url('https://gist.githubusercontent.com/netj/8836201/raw'
                             '/6f9306ad21398ea43cba4f7d537619d0e07d5ae3 '
                             '/iris.csv')
        self.assertEqual(type(df), type(pd.DataFrame()))

    def test_from_file(self):
        i = Input()
        df = i.read_from_file('ads.csv')
        self.assertEqual(type(df), type(pd.DataFrame()))


if __name__ == '__main__':
    unittest.main()
