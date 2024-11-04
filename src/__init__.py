from pathlib import Path
from sys import path

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent

path.append(str(ROOT_DIR))
