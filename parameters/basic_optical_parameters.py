from fundamentals.matlab_class import Struct
import numpy as np

params = Struct()
params.size = np.array([100, 100, 100])
params.wavelength = 0.532
params.NA = 1.2
params.RI_bg = 1.336