from constants import *

from src.utils import read_config
from src.main_tasks import main_load_db, main_tenusd
from src.model.main_model import FraudDetection

import argparse


def parser():
    parser = argparse.ArgumentParser(description='Fraud Detection and DB load tasks')
    parser.add_argument('-u', '--uuid', type=str,
                        default='44bd7ca5-7a84-41d8-a206-1a307067393c',
                        help='UUID for predict')
    parser.add_argument('-t', '--type', type=str,
                        default='all', choices=['all', 'load_db', 'model'],
                        help='Type of run')
    args = parser.parse_args()
    unpack = vars(args)
    return unpack['uuid'], unpack['type']


if __name__ == '__main__':

    uuid, type = parser()
    cfg = read_config(file_path=PATH_CFG)

    logging.info('Type = {}'.format(type))

    if type == 'all' or type == 'load_db':
        # task 1
        main_load_db(cfg=cfg,
                     db=DATABASE_URI)

        # task 2
        main_tenusd(db=DATABASE_URI,
                    cfg=cfg)
    if type == 'all' or type == 'model':
        model = FraudDetection(uuid=uuid)

        model.pipeline_train_predict()
