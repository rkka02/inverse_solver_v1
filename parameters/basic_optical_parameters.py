from fundamentals.matlab_class import Struct
from fundamentals.utils import *
import numpy as np

def basic_optical_parameters(init_params=None):
    params = Struct()
    params.size = np.array([100, 100, 100])
    params.wavelength = 0.532
    params.NA = 1.2
    params.RI_bg = 1.336
    params.resolution=np.array([0.1, 0.1, 0.1]) # resolution of one voxel
    params.vector_simulation=True # use polarised field or scalar field
    params.use_abbe_sine=True

    if init_params is not None:
        params = update_struct(params, init_params)

    return params

