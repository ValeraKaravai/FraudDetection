import pandas as pd
from constants import *
from sqlalchemy import create_engine


class EtlQuery:
    def __init__(self,
                 db,
                 cfg):
        self._db = db
        self._tables = cfg['tables']
        self._files_sql = cfg['files_sql']
        self._cfg = cfg

    def find_users_tenusd(self, verbose=True):
        file_path = self._files_sql['users_tenusd']
        users = self._execute_query(query=file_path,
                                    from_file=True,
                                    verbose=verbose)

        users_stat = [(row[0], row[1]) for row in users]

        logging.info('Count of users {}'.format(len(users_stat)))

        if verbose:
            for user in users_stat:
                logging.info('User = {}, first transaction amount_usd = {}'.format(user[0],
                                                                                   user[1]))
        users_stat = pd.DataFrame(users_stat, columns=['user_id', 'amount_usd'])

        return users_stat

    def get_table(self, tablename, query=None):

        if query is None:
            query = "SELECT * FROM {table}".format(table=tablename)

        columns_tbl = self._tables[tablename]['columns']
        df_sql = self._execute_query(query=query,
                                     from_file=False)

        df_tbl = pd.DataFrame(df_sql.fetchall(),
                              columns=columns_tbl)

        logging.info('{}. Shape = {}'.format(tablename.upper(),
                                             df_tbl.shape))

        return df_tbl

    def _execute_query(self, query, from_file=False, verbose=True):

        logging.info('Connect to {}'.format(self._db))
        engine = create_engine(self._db)

        if from_file:
            logging.info('Read query from file {}'.format(query))
            with open(query, 'r') as file:
                query = file.read()
        if verbose:
            logging.info('Execute query = {}'.format(query))

        result = engine.execute(query)

        return result
