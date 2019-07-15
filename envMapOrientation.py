import numpy as np
import math

def envMapOrientation(mEnv,nEnv, sphere_probe):
    orientationArray = np.zeros((3,mEnv, nEnv))
    for q in range(mEnv):
        for r in range(nEnv):
            if (sphere_probe):
                viewVec = orientationVectorFromSpherEnvMap(q, r, mEnv, nEnv)
            else:
                viewVec = viewVectorFrom360EnvMap(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[:,q, r] = viewVec
    return orientationArray

def orientationVectorFromSpherEnvMap(i,j,m,n):
    view = np.array([0,0,1.])
    x0 = (n-1)/2
    x = (j-x0)/x0
    y = -(i-((m-1)/2))/((m-1)/2)
    sqr = x**2 + y**2
    if (sqr > 1):
        return np.array([0,0,0])
    else:
        z = math.sqrt(1-sqr)
        normal = np.array([x,y,z])
        res = 2. * np.dot(view,normal) * normal - view
        res = np.array([res[0], res[1], res[2]])
        return res

def orientationVectorFromSpherEnvMapDebug(i,j,m,n):
    view = np.array([0,0,1.])
    x0 = (n-1)/2
    x = (j-x0)/x0
    y = -(i-((m-1)/2))/((m-1)/2)
    sqr = x**2 + y**2
    if (sqr > 1):
        return np.array([0,0,0])
    else:
        z = math.sqrt(1-sqr)
        normal = np.array([x,y,z])
        res = 2. * np.dot(view,normal) * normal - view
        res = np.array([res[0], res[1], res[2]])

    res_debug = (res + np.array([1,1,1])) / 2.
    return res_debug

def viewVectorFrom360EnvMap(i,j,m,n):
    phi = (math.pi) * i / (m-1)
    #theta = math.pi * (i-((n-1)/2))/((n-1)/2)
    #theta = 2. * math.pi * i / (n - 1)
    #theta = math.pi / 2. + 2. * math.pi * i / (n - 1)
    theta = (2. * math.pi) * (j - (n-1)/2) / (n-1)
    y = math.cos(phi)
    x = math.sin(phi) * math.sin(theta)
    z = math.sin(phi) * math.cos(theta)

    return np.array([x,y,z])

def viewVectorFrom360EnvMapDebug(i,j,m,n):
    phi = (math.pi) * i / (m-1)
    #theta = math.pi * (i-((n-1)/2))/((n-1)/2)
    #theta = 2. * math.pi * i / (n - 1)
    #theta = math.pi / 2. + 2. * math.pi * i / (n - 1)
    theta = (2. * math.pi) * (j - (n-1)/2) / (n-1)
    y = math.cos(phi)
    x = math.sin(phi) * math.sin(theta)
    z = math.sin(phi) * math.cos(theta)

    res = np.array([0,y,0])
    res_debug = (res + np.array([0,1,0])) / 2.
    return res_debug

def testInSphere(i,j,m,n):
    x0 = (n - 1) / 2.
    x = (j - x0) / x0
    y = -(i - ((m - 1) / 2.)) / ((m - 1) / 2.)
    sqr = x ** 2 + y ** 2
    if (sqr > 1):
        return False
    return True

def getMaskSphereMap(mEnv,nEnv):
    maskArray = np.zeros((mEnv, nEnv,3))
    for q in range(mEnv):
        for r in range(nEnv):
            if (testInSphere(q, r, mEnv, nEnv)):
                for k in range(3):
                    maskArray[q,r,k] = 1

    return maskArray

def envMapOrientationDebug(mEnv,nEnv, sphere_probe):
    orientationArray = np.zeros((mEnv, nEnv,3))
    for q in range((int)(mEnv/2)):
        for r in range(nEnv):
            if (sphere_probe):
                viewVec = orientationVectorFromSpherEnvMapDebug(q, r, mEnv, nEnv)
            else:
                viewVec = viewVectorFrom360EnvMapDebug(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[q, r,:] = viewVec
    return orientationArray