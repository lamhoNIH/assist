import os
import shutil

class Result:
    __path = None

    def getPath():
        if Result.__path == None:
            Result()
        return Result.__path
            
    def __init__(self, path="output", overwrite=True):
        try:
            if os.path.isdir(path):
                if overwrite:
                    print("Removing existing folder: {}".format(path))
                    shutil.rmtree(path)
                    os.makedirs(path)
            else:
                os.makedirs(path)
            Result.__path = path
        except:
            print("Creation of folder {} failed".format(path))
        else:
            print("Created folder {}".format(path))