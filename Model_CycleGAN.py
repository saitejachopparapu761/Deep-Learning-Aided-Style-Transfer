import numpy as np
from cyclegan import CycleGAN


def Model_CycleGAN(dataset, sol=None):
    if sol is None:
        sol = [0.2, 0.0002, 200]
    gan = CycleGAN(dataset, sol)
    Images = gan.train(epochs=200, batch_size=1, sample_interval=200)
    return Images
