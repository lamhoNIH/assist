import os
import shutil

class Result:
    __path = None

    def getPath():
        if Result.__path == None:
            Result()
        return Result.__path
            
    def __init__(self, path="output"):
        try:
            if os.path.isdir(path):
                print("Removing existing folder: {}".format(path))
                shutil.rmtree(path)

            os.makedirs(path, exist_ok=True)
            Result.__path = path
        except:
            print("Creation of folder {} failed".format(path))
        else:
            print("Created folder {}".format(path))