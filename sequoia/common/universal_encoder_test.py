import numpy as np
import pytest
from gym import Space, spaces
from gym.vector.utils import batch_space
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support

from .universal_encoder import create_encoder, n_parameters


@pytest.mark.slow(120)
def test_universal_encoder():
    batch_size = 10
    budget = None
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

    encoder = create_encoder(input_space, output_space, budget=budget)
    batch_input_space = batch_space(input_space, batch_size)
    batch_output_space = batch_space(output_space, batch_size)
    
    batch_input_space = add_tensor_support(batch_input_space)
    output_space = add_tensor_support(output_space)
    
    sample = batch_input_space.sample()
    encoder_output = encoder(sample)
    
    if budget:
        assert n_parameters(encoder) < budget
    
    

    