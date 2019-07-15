import numpy as np


def envMap_scale_brightness(envMap):
    perc = np.percentile(envMap, 95)
    res = envMap / perc

