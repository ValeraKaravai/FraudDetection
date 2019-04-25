import os
import sys
import logging

ROOT = os.getcwd()
sys.path.append(ROOT)

CHUNK_SIZE = 10 ** 5

# PATH
PATH_DATA = os.path.join(ROOT, 'data/')
PATH_SQL = os.path.join(ROOT, 'src/SQL/')
PATH_OUT = os.path.join(ROOT, 'output/')
PATH_CFG = os.path.join(ROOT, 'config/config.yaml')

# DATABASE
DATABASE_URI = "postgresql://localhost:5432/fraud"

# LOGGING CFG
LOG_LEVEL = logging.INFO
LOG_FORMAT = '[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s'

logging.basicConfig(format=LOG_FORMAT,
                    level=LOG_LEVEL,
                    stream=sys.stdout)
logger = logging.getLogger()


