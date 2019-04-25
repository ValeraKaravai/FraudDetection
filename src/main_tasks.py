from constants import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils import session_scope
from src.etl.etl_query import EtlQuery
from src.etl.etl_main import create_class_etl


def main_load_db(cfg, db):
    logging.info('Connect to {}'.format(db))
    engine = create_engine(db)
    session = sessionmaker(bind=engine)

    for table in cfg['tables'].keys():
        cfg_table = cfg['tables'][table]
        etl_obj = create_class_etl(table_name=table,
                                   cfg=cfg_table)
        if cfg_table['insert'] or cfg['insert_all']:
            with session_scope(session_run=session) as s:
                etl_obj.main(engine=engine,
                             sess=s,
                             drop=True)
        else:
            with session_scope(session_run=session) as s:
                etl_obj.cnt_query(sess=s)


def main_tenusd(db, cfg):
    query_obj = EtlQuery(db=db,
                         cfg=cfg)
    query_obj.find_users_tenusd()
