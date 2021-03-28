from pipeline.fit import Fit
from pipeline.validator import Validate
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from preprocess.preprocessor import Preprocessor
from algorithms.classification.decisiontreecls import DecisionTreeCls
from pipeline.input import Input
import copy

df = pd.read_csv('data/cleaned_train.csv')
pr = Preprocessor()
i = Input(file_path='data/cleaned_train.csv', target='Survived')
data = i.handle_data()


def test_removing_columns():
    data_copy = copy.deepcopy(data)
    assert len(pr.removecolumns(['Age', 'Parch'], data.X_train).columns) == len(
        data_copy.X_train.drop(['Age', 'Parch'], axis=1).columns)


def test_set_task():
    assert pr.set_task(data.y_train)[1] == 'cls'
    assert type(DecisionTreeCls()) == type(pr.set_task(data.y_train)[0][0])
    assert type(DecisionTreeCls()) == type(pr.set_task(data.y_train)[2]['DecisionTree'])
