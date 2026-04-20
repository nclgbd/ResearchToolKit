from rtk import utils
import os

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "tmp")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_CACHE_DIR, "datasets")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CACHE_DIR, "models")
LOGGING_DIR = "logs"
MAX_RAND_INT = 8192
RANDOM_STATE = 42

console = utils.get_console()
