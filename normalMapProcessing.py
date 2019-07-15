import numpy as np

def processNormalMap(normal,m,n):
    res = np.zeros((m, n, 3))
    resTemp = (normal / 255. - 0.5) * 2.
    resTemp = -resTemp
    res[:,:,0] = -resTemp[:,:,2]
    res[:, :, 1] = resTemp[:, :, 1]
    res[:, :, 2] = resTemp[:, :, 0]

    return res

def processNormalMapBis(normal,m,n):
    res = np.zeros((m, n, 3))
    resTemp = (normal / 255. - 0.5) * 2.
    return resTemp