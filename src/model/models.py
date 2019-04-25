from constants import *

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix


class Model:
    def __init__(self,
                 test_x,
                 test_y,
                 cfg,
                 train_x=None,
                 train_y=None):
        self._fit_value = cfg['model']['fit_value']
        self._cfg = cfg['model']['fit_value']
        self._mode = cfg
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._map_models = {'LogitReg': LogisticRegression,
                            'KNN': KNeighborsClassifier,
                            'LDA': LinearDiscriminantAnalysis,
                            'CART': DecisionTreeClassifier,
                            'SVM': SVC,
                            'XGB': XGBClassifier,
                            'RF': RandomForestClassifier}

    def choose_models(self):
        results = []
        names = []

        for name in self._map_models:
            model = self._map_models[name]
            k_fold = KFold(n_splits=self._fit_value['n_splits'],
                           random_state=42,
                           shuffle=True)
            cv_results = cross_val_score(estimator=model(),
                                         X=self._train_x,
                                         y=self._train_y,
                                         cv=k_fold,
                                         scoring='roc_auc')
            results.append(cv_results)
            names.append(name)
            logging.info('{}. Mean = {}, std = {}'.format(name,
                                                          round(cv_results.mean(), 4),
                                                          round(cv_results.std(), 3)))
        return results, names

    def train(self):

        params = self._fit_value['params']
        cls = self._map_models[self._fit_value['model']](**params)
        cls.fit(self._train_x, self._train_y)

        return cls

    def eval(self, cls, verbose=True):
        predicted = cls.predict(self._test_x)

        f1 = f1_score(self._test_y, predicted)
        conf_matrix = confusion_matrix(self._test_y, predicted)

        logging.info('F1 = {}'.format(f1))
        logging.info('CM = {}'.format(conf_matrix))
        if verbose:
            report = classification_report(self._test_y, predicted,
                                           output_dict=True)

            return report

    def grid_search(self, model, params):

        cls = self._map_models[model]
        grid = GridSearchCV(cls, params)
        grid.fit(self._train_x, self._train_y)
        return grid.best_estimator_, grid.best_params_
