from pathlib import Path
from PIL import Image
from src.__init__ import SRC_DIR
from src.feature_extractors.efficientnet import EfficientNet_B5
import pytest


def test_EfficientNet_B5():
    e = EfficientNet_B5(in_folder=SRC_DIR.joinpath('./test/anomalies/test_noise'))

    result = e.extract()

    assert result.shape[1:] == (456, 456)
    