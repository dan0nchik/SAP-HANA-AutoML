import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
from pipeline import Pipeline


class Model:

    def __init__(self):
        self.df = pd.DataFrame()
        self.config = None
        self.y = None
        self.X = None
        pass

    def fit(self, X_train=None, y_train=None, x_test=None, y_test=None, iterations=10, target=None, file_path=None, url=None, config=None):
        if X_train or y_train is None:
            if url is not None:
                # TODO: url validation
                data = requests.get(url).content.decode('utf-8')
                self.df = pd.read_csv(io.StringIO(data))
            if file_path is not None:
                self.df = pd.read_csv(file_path)
            self.config = config
            # TODO: exception. if no target provided, ask for it
            if not isinstance(target, list):
                target = [target]
            self.y = self.df[target]
            self.X = self.df.drop(target, axis=1)
            X_train, X_test, y_train, y_test = self.split_data()
            pipe = Pipeline(X_train, y_train, iterations=iterations)
        else:
            pipe = Pipeline(X_train, y_train, x_test, y_test, iterations=iterations)
        pipe.train()

    def split_data(self, random_state=42, test_size=0.33):
        if self.config is not None:
            test_size = self.config['test_size']
            random_state = self.config['random_state']
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=random_state,
                                                            test_size=test_size)
        return X_train, X_test, y_train, y_test

# i = BaseEstimator(target='y', file_path='bank.csv')
# 
# print(i.split_data()[0].head())
# #       age          job   marital  education  ... campaign  pdays previous poutcome
# # 1138   35   technician  divorced  secondary  ...       12     -1        0  unknown
# # 209    56      retired   married  secondary  ...       14     -1        0  unknown
# 
# i_url = BaseEstimator(target='Value', url='https://www.stats.govt.nz/assets/Uploads/Annual-enterprise-survey/Annual'
#                                   '-enterprise-survey-2019-financial-year-provisional/Download-data/annual-enterprise'
#                                   '-survey-2019-financial-year-provisional-csv.csv')
# print(i_url.split_data()[2].head)
# # 27439        9
# # 13267      165
# # 31279      133
# # 11794      262
# #          ...
