from pathlib import Path
from PIL import Image
from src.__init__ import SRC_DIR
from src.feature_extractors.efficientnet import EfficientNet_B5, EfficientNet_V2_L



def test_EfficientNet_B5():
    e = EfficientNet_B5(in_folder=SRC_DIR.joinpath('./test/anomalies/test_noise'))

    result = e.extract()
    del e

    assert len(result.shape) == 4
    assert result.shape[1:] == (2048, 15, 15)



def test_EfficientNet_V2_L():
    e = EfficientNet_V2_L(in_folder=SRC_DIR.joinpath('./test/anomalies/test_noise'))

    result = e.extract()
    del e

    assert len(result.shape) == 4
    assert result.shape[1:] == (1280, 15, 15)
