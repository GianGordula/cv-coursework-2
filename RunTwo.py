import numpy as np 
from PIL import Image, ImageOps
import glob


resourcePath = "./resources/extractedImages"
trainingImages = resourcePath + "/training"
testingImages = resourcePath + "/testing"

def getTrainingPaths():
    return glob.glob(trainingImages + "/*/*.jpg")
     

def loadImage(image: Image) -> np.array:
    return np.array(image)

def normalise(patch) -> np.array:
    if(np.std(patch) == 0):
        return patch
    return patch/np.std(patch)

def center(patch) -> np.array:
    mean = np.mean(patch)
    map(lambda a : a - mean, patch)
    return patch



#TODO::flatten the patches, and rest of bollocks
def getPatches(image, step, width, height):
    patches = np.empty(shape=(0, width, height), dtype=np.int32)
    for y in range(0, image.shape[0] - height + 1, step):
        for x in range(0, image.shape[1] - width + 1, step):
            patch = image[y:y+height,x:x + width]
            patch = normalise(patch)
            patch = center(patch)
            patch = np.reshape(patch, (1,width,height))
            patches = np.append(patches,patch, axis = 0)
    return patches


tester = np.array(Image.open(getTrainingPaths()[0]))
print(tester.shape)
print(getPatches(tester, 4, 8, 8).shape)


