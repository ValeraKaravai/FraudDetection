from src.model.models import Model
from src.model.data_visualize import visualize_roc_score
from src.model.data_preprocessing import DataPreprocessing
from src.utils import load_pkl, save_pkl

from src.utils import read_config
from constants import *
import pandas as pd


class FraudDetection:
    def __init__(self, uuid=None):
        self._uuid = uuid
        self._cfg = read_config(file_path=PATH_CFG)
        self._mode = self._cfg['model']['mode']
        self._logging = self._cfg['log']

        self._preproc_obj = DataPreprocessing(cfg=self._cfg,
                                              db=DATABASE_URI)
        self._actions = {'LOCK': 'LOCK_USER',
                         'ALERT': 'ALERT_AGENT',
                         'BOTH': 'BOTH',
                         'NOTHING': 'NON FRAUDSTER'}
        self._threshold_amount = 1000
        self._threshold_prob_suspect = 0.5
        self._threshold_prob_alert = 0.6
        self._threshold_prob_lock = 0.7

        if not self._logging:
            logger.setLevel(logging.ERROR)

    def pipeline_choose_sampling(self, users, transactions, types_sampling):

        for sample in types_sampling:
            train_x, train_y, test_x, test_y, *_ = self._preproc_obj.main_features_selection(type_sampling=sample,
                                                                                             users=users,
                                                                                             transactions=transactions)
            model_obj = Model(train_x=train_x,
                              train_y=train_y,
                              test_x=test_x,
                              test_y=test_y, cfg=self._cfg)
            results, names = model_obj.choose_models()

            visualize_roc_score(df=results,
                                names=names,
                                type_sampl=sample)

    def pipeline_train_predict(self):
        if self._mode == 'train':
            self.train_mode()
        if self._mode == 'predict':
            self.predict_mode()

    def get_action(self, users, probability):
        probability = probability[0][1]
        has_transactions = users.has_transactions[0]
        mean_amount = users.mean_amount[0]
        is_lock = users.is_lock[0]
        is_minos = users.source_minos[0]
        kyc = users.kyc[0]

        action_user = 'NOTHING'

        if has_transactions == 0:
            return action_user

        if probability >= self._threshold_prob_lock:
            return 'BOTH'

        if probability >= self._threshold_prob_alert:
            if (is_lock == 1) or \
                    (mean_amount >= self._threshold_amount) or (kyc == 'failed'):
                return 'BOTH'
            elif probability >= self._threshold_prob_suspect:
                return 'ALERT'

        if probability >= self._threshold_prob_suspect:
            if kyc != 'SUCCESS' or is_minos == 1:
                return 'ALERT'

        return action_user

    def train_mode(self):
        fit_value = self._cfg['model']['fit_value']

        train_x, train_y, test_x, test_y, *_ = self._preproc_obj.main_features_selection(
            type_sampling=fit_value['sampling'])

        model_obj = Model(train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          cfg=self._cfg)
        cls = model_obj.train()
        report = model_obj.eval(cls=cls)
        logging.info('{}'.format(pd.DataFrame(report)))
        save_pkl(filename=self._cfg['model']['files']['model'],
                 obj=cls)

        self.predict_mode()

    def predict_mode(self):

        try:
            cls = load_pkl(filename=self._cfg['model']['files']['model'])
        except OSError:
            logging.error(
                'File of model not found {}. Please, fix mode of train'.format(self._cfg['model']['files']['model']))
            sys.exit()
        if self._uuid is None:
            logging.info('UUID is None')
            return None
        logging.info('Start predict for {}'.format(self._uuid))
        column_feature = load_pkl(filename=self._cfg['model']['files']['columns'])
        x_scale_set, y_set, users = self._preproc_obj.feature_selection_uuid(uuid=self._uuid,
                                                                             columns=column_feature)

        prob = cls.predict_proba(x_scale_set)
        prediction = cls.predict(x_scale_set)

        action = self.get_action(users=users,
                                 probability=prob)

        logging.info('Prob = {}, class = {}'.format(prob, prediction))
        logging.info('Action = {}'.format(self._actions[action]))
        print(self._actions[action])
        return prob
