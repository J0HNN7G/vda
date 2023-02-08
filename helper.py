import numpy as np


def newScale(oldValue, oldMin=0, oldMax=1, newMin=-1, newMax=1):
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    return newValue
