from .universal_encoder import create_encoder
from gym import Space, spaces
from gym.vector.utils import batch_space
import numpy as np

from common.gym_wrappers.convert_tensors import wrap_space
from .universal_encoder import n_parameters

def test_universal_encoder():
    batch_size = 10
    input_space = spaces.Dict({
        "x": spaces.Box(low=0, high=1, shape=[3, 32, 32]),
        "t": spaces.Discrete(2),
    })
    output_space = spaces.Box(
        -np.inf,
        np.inf,
        shape=[512,],
        dtype=np.float32,
    )

    encoder = create_encoder(input_space, output_space, budget=None)
    batch_input_space = batch_space(input_space, batch_size)
    batch_output_space = batch_space(output_space, batch_size)
    
    batch_input_space = wrap_space(batch_input_space)
    output_space = wrap_space(output_space)
    
    sample = batch_input_space.sample()
    encoder_output = encoder(sample)
    assert False, n_parameters(encoder)
    
    

    