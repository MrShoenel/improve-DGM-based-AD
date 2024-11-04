from pathlib import Path
from PIL import Image
from src.__init__ import SRC_DIR
from src.feature_extractors.convnext import ConvNextV2


def test_ConvNextV2():
    c = ConvNextV2(in_folder=SRC_DIR.joinpath('./test/anomalies/test_noise')).eval()

    result = c.extract()
    del c

    assert len(result.shape) == 4
    assert result.shape[1:] == (2816, 16, 16)

