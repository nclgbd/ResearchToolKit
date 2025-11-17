import os

__version__ = "0.1.0"

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "tmp")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_CACHE_DIR, "datasets")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CACHE_DIR, "models")

RANDOM_STATE = 42
MAX_RAND_INT = 8192
