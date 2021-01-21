import numpy as np

def foundTwoHils (array):
    arrayDiff = -np.diff(np.r_[array[1], array[1:]])
    tempMax = []
    maxInd = []

    for rad in np.arange(0, len(array) - 50, 50):
        tempMax.append(arrayDiff[rad:rad + 50].max())
        maxInd.append(arrayDiff[rad:rad + 50].argmax(axis=0) + rad)
    # First Maxima
    tmepMaxSoretd = np.sort(tempMax)
    tempIndSorted = np.argsort(tempMax)
    localMaximasInds = [maxInd[tempIndSorted[-1]]]
    itterFlag = True
    ind = 2
    while itterFlag and ind < len(tempIndSorted):
        if tempIndSorted[-1] - tempIndSorted[-ind] > 2:
            localMaximasInds.append(maxInd[tempIndSorted[-ind]])
            itterFlag = False
        else:
            ind += 1
    localMaximasInds = localMaximasInds[::-1]

    return localMaximasInds