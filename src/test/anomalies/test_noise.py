from os import environ
environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform' # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

from src.anomalies.noise import Perlin_generator_JAX
from pathlib import Path
import jax.numpy as jnp
from PIL import Image
import pytest



@pytest.mark.parametrize('seed', [1,2,3,4,5])
def test_augment_with_mask_and_texture(seed: int):
    """
    Let's test creating a LAS-style anomaly using the images nominal, fg, and texture{.png}.
    """
    
    THIS_DIR = Path(__file__).parent
    IMAGE_DIR = THIS_DIR.joinpath('./test_noise')

    pgj = Perlin_generator_JAX(seed=seed)


    nominal = pgj.open_hwc_asfloat(IMAGE_DIR.joinpath('./nominal.png'))
    pgj.save_chw_from_float(nominal, IMAGE_DIR.joinpath(f'./nominal_repl_{seed}.png'))

    texture = pgj.open_hwc_asfloat(IMAGE_DIR.joinpath('./texture.png'))
    pgj.save_chw_from_float(texture, IMAGE_DIR.joinpath(f'./texture_repl_{seed}.png'))

    fg = jnp.ceil(jnp.asarray(Image.open(IMAGE_DIR.joinpath('./fg.png'))) / 255.)
    assert fg.dtype == jnp.float32 and jnp.all(fg <= 1.0)

    mask_s, mask_l = pgj.perlin_mask(img_shape=nominal.shape, feat_size=576//8, min=0, max=6, mask_fg=fg)
    pgj.save_chw_from_float(mask_l, IMAGE_DIR.joinpath(f'./mask_{seed}.png'))
    
    augmented = pgj.augment(nom=nominal, tex=texture, mask=mask_l, seed=seed + 1)
    pgj.save_chw_from_float(augmented, IMAGE_DIR.joinpath(f'./augmented_{seed}.png'))

