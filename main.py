from models import Model
import pandas as pd
from sklearn.model_selection import train_test_split

cls = Model()
cls.automl(file_path='data/reg.csv', target='Все 18+_TVR')

