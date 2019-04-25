import time
import pandas as pd
from constants import *

from src.etl.models import Base, User, Transaction, \
    FxRates, CurrencyDetails


class EtlMain:
    '''
    Main class for ETL scripts (for create DB and insert of datasets)
    '''

    def __init__(self,
                 cfg,
                 table_name,
                 chunksize=CHUNK_SIZE,
                 sep=','):
        self._file_path = cfg['file']
        self._extra_path = cfg['extra_file']
        self._table_name = table_name
        self._sep = sep
        self._columns = cfg['columns']
        self._chunksize = chunksize

    def load_data(self, extra=False):
        file_path = self._file_path
        if extra:
            file_path = self._extra_path

        df_main_list = []

        time_start = time.time()

        logging.info('{}. Read file {} by chunk = {}'.format(self._table_name.upper(),
                                                             file_path,
                                                             self._chunksize))
        for chunk in pd.read_csv(file_path,
                                 sep=self._sep,
                                 chunksize=self._chunksize):
            chunk = chunk.replace({pd.np.nan: None})

            df_main_list.append(chunk)

        df_main = pd.concat(df_main_list)

        logging.info('{}. Convert names of columns to lowercase'.format(self._table_name.upper()))
        df_main.rename(str.lower,
                       axis='columns',
                       inplace=True)

        time_end = round(time.time() - time_start, 2)
        logging.info('{}. Read file, time = {} sec, shape = {}'.format(self._table_name.upper(),
                                                                       time_end,
                                                                       df_main.shape))

        return df_main

    def columns(self, df):
        return df[self._columns]

    def create_table(self, engine):
        if not engine.dialect.has_table(engine,
                                        table_name=self._table_name):
            logging.info('{}. Create table'.format(self._table_name.upper()))
            Base.metadata.tables[self._table_name].create(engine)
        else:
            logging.info('{}. Table exists'.format(self._table_name.upper()))

    def drop_table(self, engine):

        if engine.dialect.has_table(engine,
                                    table_name=self._table_name):
            logging.info('{}. Drop table'.format(self._table_name.upper()))
            Base.metadata.tables[self._table_name].drop(engine)
        else:
            logging.info('{}. Table does not exist'.format(self._table_name.upper()))

    def insert(self, sess, df):

        cls = self.get_class_tbl()

        logging.info('{}. Insert {} rows'.format(self._table_name.upper(),
                                                 df.shape))

        time_start = time.time()

        for i in range(0, df.shape[0] + self._chunksize, self._chunksize):
            df_sub = df.iloc[i:(i + self._chunksize), :]
            sess.bulk_insert_mappings(mapper=cls,
                                      mappings=df_sub.to_dict(orient="records"),
                                      render_nulls=True, return_defaults=True)

        time_end = round(time.time() - time_start, 2)
        logging.info('{}. Insert time = {} sec'.format(self._table_name.upper(),
                                                       time_end))

        sess.commit()

    def cnt_query(self, sess):

        cls = self.get_class_tbl()
        cnt_rows = sess.query(cls).count()
        logging.info('{}. Cnt row = {}'.format(self._table_name.upper(),
                                               cnt_rows))
        return cnt_rows

    def main(self, engine, sess, drop=False):

        if drop:
            self.drop_table(engine=engine)

        self.create_table(engine=engine)

        df = self.prepare()

        self.insert(sess=sess,
                    df=df)
        cnt_rows = self.cnt_query(sess=sess)

        if cnt_rows == df.shape[0]:
            logging.info('{}. Successful insert!'.format(self._table_name.upper()))
        else:
            logging.info('{}. Not success {} {}'.format(self._table_name.upper(),
                                                        cnt_rows,
                                                        df.shape[0]))

        return

    @staticmethod
    def drop_all(engine):
        logging.info('Drop table')
        Base.metadata.drop_all(engine)

    @staticmethod
    def create_all(engine):
        logging.info('Create table')
        Base.metadata.create_all(engine)

    @staticmethod
    def get_class_tbl():
        return logging.info('Main class')

    def prepare(self):
        return logging.info('Main class')


class TransactionsData(EtlMain):
    '''
    Class for preprocessing dataset of transactions.
    Convert merchant country (code3 and digits) to code2 (because users country in code2)
    '''

    def prepare(self):
        df = self.load_data()

        df_country = self.load_data(extra=True)

        cnt_len3 = sum(df.merchant_country.str.len() > 3)
        logging.info('{}. Merchant country with len > 3: {}'.format(self._table_name.upper(),
                                                                    cnt_len3))
        logging.info('{}. Get last 3 symbol from merchant_country'.format(self._table_name.upper()))

        df.merchant_country = df.merchant_country.str[-3:].str.upper().str.strip()

        logging.info('{}. Convert to code2 from code3 and digits'.format(self._table_name.upper()))

        df.merchant_country = self.__prepare_merchant_country(df_country=df_country,
                                                              merchant_country=df.merchant_country.copy())

        return self.columns(df=df)

    def __prepare_merchant_country(self, df_country, merchant_country):
        df_country.dropna(inplace=True)
        df_country.code3 = df_country.code3.str.upper()
        df_country.code = df_country.code.str.upper()

        logging.info('{}. Create map dictionary'.format(self._table_name.upper()))
        code = pd.Series(df_country.code.values,
                         index=df_country.code3).to_dict()

        digits = pd.Series(df_country.code.values,
                           index=df_country.numcode.astype(int).astype(str)).to_dict()

        map_codes = {**code, **digits}

        logging.info('{}. Remove the first zero in digits (ex. 001=1)'.format(self._table_name.upper()))
        ind_digit = merchant_country.str.isdigit() == True
        int_value = merchant_country[ind_digit].astype(int).astype(str)
        merchant_country[ind_digit] = int_value

        logging.info('{}. Map merchant country to new value'.format(self._table_name.upper()))
        merchant_country.replace(map_codes,
                                 inplace=True)

        return merchant_country

    @staticmethod
    def get_class_tbl():
        return Transaction


class CurrencyDetailsData(EtlMain):
    '''
    Class for preprocessing dataset of currency.
    '''

    def prepare(self):
        df = self.load_data()

        df.rename(columns={'currency': 'ccy'},
                  inplace=True)
        return self.columns(df=df)

    @staticmethod
    def get_class_tbl():
        return CurrencyDetails


class UsersData(EtlMain):
    '''
    Class for preprocessing dataset of users.
    Add column `is_fraudster`
    '''

    def prepare(self):
        df = self.load_data()

        df_fraud = self.load_data(extra=True).user_id

        df['is_fraudster'] = df.id.isin(df_fraud)

        return self.columns(df=df)

    @staticmethod
    def get_class_tbl():
        return User


class FxRatesData(EtlMain):
    '''
    Class for preprocessing dataset of fx rates.
    '''

    def prepare(self):
        df = self.load_data()

        df.rename(columns={'unnamed: 0': 'ts'},
                  inplace=True)

        df.set_index('ts',
                     drop=True,
                     inplace=True)

        tt, base_ccy, ccy = self.parse_ccy(columns=df.columns)

        df.columns = tt

        df = df.stack().reset_index()
        df.rename(columns={0: 'rate'},
                  inplace=True)
        df.rate = abs(df.rate)
        df['base_ccy'] = df.level_1.map(base_ccy)
        df['ccy'] = df.level_1.map(ccy)

        return self.columns(df=df)

    @staticmethod
    def get_class_tbl():
        return FxRates

    @staticmethod
    def parse_ccy(columns):
        columns = [i.upper() for i in columns]
        concat_ccy = [i[0:3] + '_' + i[3:] for i in columns]

        base_ccy = {}
        ccy = {}

        for i in concat_ccy:
            i_split = i.split('_')
            base_ccy.update({i: i_split[0]})
            ccy.update({i: i_split[1]})

        return concat_ccy, base_ccy, ccy


def create_class_etl(**args):
    mapping_class = {'users': UsersData,
                     'transactions': TransactionsData,
                     'currency_details': CurrencyDetailsData,
                     'fx_rates': FxRatesData}

    return mapping_class[args['table_name']](**args)
