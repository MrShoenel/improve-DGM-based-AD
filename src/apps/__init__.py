from pathlib import Path
APPS_DIR = Path(__file__).parent
SRC_DIR = APPS_DIR.parent
ROOT_DIR = SRC_DIR.parent
from sys import path
path.append(str(ROOT_DIR))
import src.__init__
