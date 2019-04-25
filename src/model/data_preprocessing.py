import pandas as pd
from constants import *

from src.etl.etl_query import EtlQuery
from src.utils import save_pkl, load_pkl

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.model.data_visualize import visualize_tsn_pca


class DataPreprocessing(EtlQuery):
    map_sampling = {'under': RandomUnderSampler(random_state=42),
                    'smote': SMOTE(random_state=42),
                    'combine': SMOTEENN(random_state=42)}

    map_decompose = {'pca': PCA(n_components=2, random_state=42),
                     'tsne': TSNE(n_components=2, random_state=42)}

    def convert_transactions(self, is_fraudster, uuid=None):

        if uuid is None:
            transactions = self.get_table(tablename='transactions')
        else:
            transactions = self.get_table(tablename='transactions',
                                          query="SELECT * FROM transactions WHERE user_id = '{}'".format(uuid))

        if transactions.shape[0] == 0:
            transactions['amount_usd'] = None
            transactions['first_amount'] = None
            transactions['first_date'] = None
            transactions['is_fraudster'] = None
            return transactions

        fx_rates = self.get_table(tablename='fx_rates')
        currency_details = self.get_table(tablename='currency_details')

        logging.info('TRANSACTIONS. Convert to usd, amount_usd')

        transactions['amount_usd'] = self.convert_to_usd(fx_rates=fx_rates.copy(),
                                                         transactions=transactions.copy(),
                                                         currency_details=currency_details.copy())

        logging.info('TRANSACTIONS. Create first_amount, first_date features')
        transactions['first_amount'], \
        transactions['first_date'] = self.get_first_transactions(transactions=transactions)

        transactions['is_fraudster'] = transactions.user_id.isin(is_fraudster)

        return transactions

    def convert_user(self, uuid=None):

        if uuid is None:
            users = self.get_table(tablename='users')
        else:
            users = self.get_table(tablename='users',
                                   query="SELECT * FROM users WHERE id = '{}'".format(uuid))
        if users.shape[0] == 0:
            logging.error('Users is not found')
            sys.exit()
        users['created_date_origin'] = users.created_date
        logging.info('USERS. Convert date to numeric (days)')

        users.created_date, min_date = self.convert_date_to_numeric(dates=users.created_date)

        users['terms_version_origin'] = users.terms_version.astype(str)

        logging.info('USERS. Rename STATE to state_user')
        users['is_lock'] = 1 * (users.state == 'LOCK')
        users.drop('state',
                   axis=1,
                   inplace=True)

        return users, min_date

    def convert_main(self, uuid=None):

        na_cfg = self._cfg['model']['data_value']['na']

        logging.info('USERS. Convert data')
        users, min_date = self.convert_user(uuid=uuid)

        logging.info('TRANSACTIONS. Convert data')
        transactions = self.convert_transactions(is_fraudster=users.id[users.is_fraudster],
                                                 uuid=uuid)

        logging.info('FEATURE TABLE. Convert has_email to int')
        users['has_email'] = 1 * users.has_email

        logging.info('FEATURE TABLE. Create first_amount, first_date for users')
        users['first_amount'], users['first_date'] = self.get_first_transactions_users(users=users,
                                                                                       transactions=transactions)
        logging.info('FEATURE TABLE. Create country_gb (bool-int)')
        users['country_gb'] = 1 * (users.country == 'GB')

        logging.info('FEATURE TABLE. Create diff date (first_transactions - created date)')

        if users.first_date.isnull().all():
            users['diff_date'] = None
        else:
            users['diff_date'] = (users.first_date - users.created_date_origin).dt.total_seconds()

        logging.info('FEATURE TABLE. Create first_success (bool-int)')
        users['first_success'] = 1 * (users.first_amount > 0)

        for i in ['source', 'type', 'state']:
            logging.info('FEATURE TABLE. Create {}_freq'.format(i))
            users[i + '_freq'] = self.frequency_level(users=users,
                                                      transactions=transactions,
                                                      variable=i)
        logging.info('FEATURE TABLE. Create source_minos (bool-int)')
        if users.source_freq.isnull().all():
            users['source_minos'] = 0
        else:
            users['source_minos'] = 1 * (users.source_freq == 'MINOS')

        logging.info('FEATURE TABLE. Create is_user_country, has_transactions, cnt_currency, mean_amount (bool-int)')
        users = self.get_user_transaction_feature(users=users,
                                                  transactions=transactions)

        # logging.info('FEATURE TABLE. Info {}'.format(users.info()))
        users = self.make_fill_na(df=users,
                                  cfg=na_cfg)

        return users, transactions

    def prepare_data(self, users, columns=None):

        cfg_data_value = self._cfg['model']['data_value']

        features = cfg_data_value['features_dummy'] + \
                   cfg_data_value['features_not_dummy']

        label = cfg_data_value['label']

        logging.info('PREPROCESSING. Get features {}'.format(features))

        y_set = 1 * users[label]

        x_set = users[features]

        logging.info('PREPROCESSING. Dummy')
        x_set_dummy, x_scale_set = self.dummy_scale_preproc(x_set=x_set,
                                                            column_dummy=cfg_data_value['features_dummy'],
                                                            columns=columns)

        return x_scale_set, y_set, x_set_dummy.columns

    def get_test_train(self, x_scale_set, y_set, cfg_data_value, type_sampling):

        logging.info('PREPROCESSING. Split test train')
        train_x, test_x, train_y, test_y = train_test_split(x_scale_set,
                                                            y_set,
                                                            random_state=42,
                                                            test_size=cfg_data_value['test_size'],
                                                            stratify=y_set)
        if type_sampling is not None:
            logging.info('PREPROCESSING. Sampling = {}'.format(type_sampling))

            train_x, train_y = self.imbalance_sampling(x_set=train_x,
                                                       y_set=train_y.iloc[:, 0],
                                                       sampling_obj=self.map_sampling[type_sampling])

        return train_x, train_y, test_x, test_y

    def main_features_selection(self,
                                type_sampling=None,
                                users=None, transactions=None):

        if users is None or transactions is None:
            users, transactions = self.convert_main()

        x_scale_set, y_set, column_feature = self.prepare_data(users=users)

        save_pkl(filename=self._cfg['model']['files']['columns'],
                 obj=column_feature)

        train_x, train_y, test_x, test_y = self.get_test_train(x_scale_set=x_scale_set,
                                                               y_set=y_set,
                                                               cfg_data_value=self._cfg['model']['data_value'],
                                                               type_sampling=type_sampling)

        return train_x, train_y, test_x, test_y, x_scale_set, y_set, users, transactions

    def feature_selection_uuid(self, uuid, columns):
        users, transactions = self.convert_main(uuid=uuid)

        x_scale_set, y_set, column_feature = self.prepare_data(users=users,
                                                               columns=columns)

        return x_scale_set, y_set, users

    def dummy_scale_preproc(self, x_set, column_dummy, columns):

        logging.info('PREPROCESSING. Get dummy {}'.format(column_dummy))
        x_set = pd.get_dummies(x_set, columns=column_dummy)

        if columns is not None:
            x_set = x_set.reindex(columns=columns, fill_value=-1)
        x_set = x_set.astype(float)
        logging.info('PREPROCESSING. Scalar X set')

        if self._cfg['model']['mode'] != 'train':
            scaler = load_pkl(filename=self._cfg['model']['files']['scaler'])
            x_scale_set = scaler.transform(x_set)
        else:
            scaler = StandardScaler()
            x_scale_set = scaler.fit_transform(x_set)

            save_pkl(filename=self._cfg['model']['files']['scaler'],
                     obj=scaler)

        return x_set, x_scale_set

    @staticmethod
    def get_user_transaction_feature(users, transactions):
        user_transactions = pd.merge(users, transactions,
                                     left_on=['id'],
                                     right_on=['user_id'],
                                     how='left', suffixes=['_x', '_y'])
        user_transactions['is_user_country'] = 1 * (user_transactions.country == user_transactions.merchant_country)

        user_transactions['has_transactions'] = 1 * user_transactions.user_id.notnull()

        user_transactions.amount_usd = user_transactions.amount_usd.astype(float)
        users_features = user_transactions.groupby('id_x').agg({'is_user_country': ['sum'],
                                                                'has_transactions': ['sum'],
                                                                'currency': ['nunique'],
                                                                'amount_usd': ['mean']}).reset_index()
        users_features.columns = ['user_id', 'is_user_country', 'has_transactions', 'cnt_currency', 'mean_amount']

        users_features['is_user_country'] = 1 * users_features['is_user_country'].astype(bool)
        users_features['has_transactions'] = 1 * users_features['has_transactions'].astype(bool)

        merge_df = pd.merge(users,
                            users_features,
                            left_on='id',
                            right_on='user_id',
                            how='inner')
        return merge_df

    @staticmethod
    def frequency_level(users, transactions, variable):
        group_cnt = transactions.groupby(by=['user_id', variable])['id'].agg(
            [('cnt', 'count')]).sort_values(
            by=['user_id', 'cnt'], ascending=False).reset_index()

        freq_stat = group_cnt.groupby('user_id').first().reset_index()[['user_id', variable]]
        users_variable = pd.merge(users, freq_stat,
                                  left_on='id',
                                  right_on='user_id',
                                  how='left')
        return users_variable[variable]

    @staticmethod
    def get_first_transactions_users(transactions, users):
        df_merge = pd.merge(users,
                            transactions[transactions.first_date.notnull()][
                                ['user_id', 'first_amount', 'first_date']],
                            left_on='id', right_on='user_id', how='left')[
            ['first_amount', 'first_date']]
        return df_merge.first_amount, df_merge.first_date

    @staticmethod
    def get_first_transactions(transactions):
        df_min = transactions.groupby('user_id')['created_date'].min().reset_index()
        df_min.columns = ['user_min', 'date_min']
        df_min_merge = pd.merge(transactions, df_min,
                                left_on=['user_id', 'created_date'],
                                right_on=['user_min', 'date_min'],
                                how='left')

        df_min_merge['first_amount'] = df_min_merge.amount_usd
        df_min_merge.loc[(df_min_merge.user_min.isnull()) | (df_min_merge.state != 'COMPLETED'), 'first_amount'] = None
        return df_min_merge.first_amount, df_min_merge.date_min

    @staticmethod
    def convert_date_to_numeric(dates, min_date=None):
        if min_date is None:
            min_date = min(dates)
        timedeltas = (dates - min_date)

        return timedeltas.apply(lambda x: x.days), min_date

    @staticmethod
    def make_fill_na(df, cfg):

        for column in cfg:

            if column == 'default':
                continue

            logging.info('Fill_na column: {}, value: {}'.format(column,
                                                                cfg[column]))
            df[column].fillna(value=cfg[column], inplace=True)

        logging.info('Fill na other columns = {}'.format(cfg['default']))
        df.fillna(cfg['default'], inplace=True)

        return df

    @staticmethod
    def convert_to_usd(fx_rates, currency_details, transactions):
        fx_rates.ts = fx_rates.ts.dt.round("D")
        fx_rates = fx_rates[fx_rates.base_ccy == 'USD']
        fx_rates_gr = fx_rates.groupby(by=['ts', 'ccy'])['rate'].mean().reset_index()

        transactions.created_date = transactions.created_date.dt.round("D")
        merge_rate = pd.merge(transactions, fx_rates_gr,
                              left_on=['created_date', 'currency'],
                              right_on=['ts', 'ccy'],
                              how='left')

        merge_rate.loc[merge_rate.currency == 'USD', 'rate'] = 1

        df_amount = pd.merge(merge_rate, currency_details,
                             left_on=['currency'],
                             right_on=['ccy'],
                             how='inner')

        df_amount['amount_usd'] = df_amount.rate * df_amount.amount / pow(10, df_amount.exponent)

        return df_amount['amount_usd']

    @staticmethod
    def imbalance_sampling(x_set, y_set, sampling_obj):

        logging.info('Shape before = {}, {}'.format(x_set.shape, y_set.shape))

        x_sample, y_sample = sampling_obj.fit_resample(x_set, y_set)

        logging.info('Shape after x = {}, y = {} '.format(x_sample.shape, y_sample.shape))

        return x_sample, y_sample

    def get_decompose(self, x_set, y_set, type_decompose):

        decompose = self.map_decompose[type_decompose]
        df_decompose = decompose.fit_transform(x_set)

        visualize_tsn_pca(df_decompose=df_decompose,
                          y_set=y_set,
                          type=type_decompose)
