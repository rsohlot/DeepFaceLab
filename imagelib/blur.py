import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import line

class LineDictionary:
    def __init__(self):
        self.lines = {}
        self.Create3x3Lines()
        self.Create5x5Lines()
        self.Create7x7Lines()
        self.Create9x9Lines()
        return
    
    def Create3x3Lines(self):
        lines = {
            0: [1, 0, 1, 2],
            45: [2, 0, 0, 2],
            90: [0, 1, 2, 1],
            135: [0, 0, 2, 2],
        }

        self.lines[3] = lines
        return
    
    def Create5x5Lines(self):
        lines = {
            0: [2, 0, 2, 4],
            22.5: [3, 0, 1, 4],
            45: [0, 4, 4, 0],
            67.5: [0, 3, 4, 1],
            90: [0, 2, 4, 2],
            112.5: [0, 1, 4, 3],
            135: [0, 0, 4, 4],
            157.5: [1, 0, 3, 4],
        }

        self.lines[5] = lines
        return
        
    def Create7x7Lines(self):
        lines = {
            0: [3, 0, 3, 6],
            15: [4, 0, 2, 6],
            30: [5, 0, 1, 6],
            45: [6, 0, 0, 6],
            60: [6, 1, 0, 5],
            75: [6, 2, 0, 4],
            90: [0, 3, 6, 3],
            105: [0, 2, 6, 4],
            120: [0, 1, 6, 5],
            135: [0, 0, 6, 6],
            150: [1, 0, 5, 6],
            165: [2, 0, 4, 6],
        }

        self.lines[7] = lines
        return
    
    def Create9x9Lines(self):
        lines = {
            0: [4, 0, 4, 8],
            11.25: [5, 0, 3, 8],
            22.5: [6, 0, 2, 8],
            33.75: [7, 0, 1, 8],
            45: [8, 0, 0, 8],
            56.25: [8, 1, 0, 7],
            67.5: [8, 2, 0, 6],
            78.75: [8, 3, 0, 5],
            90: [8, 4, 0, 4],
            101.25: [0, 3, 8, 5],
            112.5: [0, 2, 8, 6],
            123.75: [0, 1, 8, 7],
            135: [0, 0, 8, 8],
            146.25: [1, 0, 7, 8],
            157.5: [2, 0, 6, 8],
            168.75: [3, 0, 5, 8],
        }

        self.lines[9] = lines
        return
        
lineLengths =[3,5,7,9]
lineTypes = ["full", "right", "left"]

lineDict = LineDictionary()

def LinearMotionBlur_random(img):
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    return LinearMotionBlur(img, lineLength, lineAngle, lineType)

def LinearMotionBlur(img, dim, angle, linetype='full'):
    if len(img.shape) == 2:
        h, w = img.shape
        c = 1
        img = img[...,np.newaxis]
    elif len(img.shape) == 3:
        h,w,c = img.shape
    else:
        raise ValueError('unsupported img.shape')
    
    kernel = LineKernel(dim, angle, linetype)
    
    imgs = []    
    for i in range(c):
        imgs.append ( convolve2d(img[...,i], kernel, mode='same') )
    
    img = np.stack(imgs, axis=-1)
    img = np.squeeze(img) 
    return img

def LineKernel(dim, angle, linetype):
    kernelwidth = dim
    kernelCenter = int(math.floor(dim/2))
    angle = SanitizeAngleValue(kernelCenter, angle)
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    lineAnchors = lineDict.lines[dim][angle]
    if linetype == 'left':
        lineAnchors[2] = kernelCenter
        lineAnchors[3] = kernelCenter
    elif linetype == 'right':
        lineAnchors[0] = kernelCenter
        lineAnchors[1] = kernelCenter
    rr,cc = line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
    kernel[rr,cc]=1
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel

def SanitizeAngleValue(kernelCenter, angle):
    numDistinctLines = kernelCenter * 4
    angle = math.fmod(angle, 180.0)
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angle = nearestValue(angle, validLineAngles)
    return angle

def nearestValue(theta, validAngles):
    idx = (np.abs(validAngles-theta)).argmin()
    return validAngles[idx]

def randomAngle(kerneldim):
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])