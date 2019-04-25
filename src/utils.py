import yaml
import pickle
from constants import *

from contextlib import contextmanager


def read_config(file_path):
    with open(file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    cfg = generate_path(cfg=cfg)
    return cfg


def generate_path(cfg):
    for model_file in cfg['model']['files']:
        cfg['model']['files'][model_file] = os.path.join(PATH_OUT,
                                                         cfg['model']['files'][model_file])

    for file_sql in cfg['files_sql']:
        cfg['files_sql'][file_sql] = os.path.join(PATH_SQL,
                                                  cfg['files_sql'][file_sql])

    for file_data in cfg['tables']:
        cfg['tables'][file_data]['file'] = os.path.join(PATH_DATA,
                                                        cfg['tables'][file_data]['file'])
        cfg['tables'][file_data]['extra_file'] = os.path.join(PATH_DATA,
                                                              cfg['tables'][file_data]['extra_file'])

    return cfg


@contextmanager
def session_scope(session_run):
    session = session_run()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_pkl(filename, obj):
    logging.info('Save obj to {}'.format(filename))
    pickle.dump(obj, open(filename, 'wb'))


def load_pkl(filename):
    return pickle.load(open(filename, 'rb'))
