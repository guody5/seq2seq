import numpy as np
import time
import encoder
import decoder

class Config(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000