import os
from sys import platform

# To use this class properly, specify the path in the constructor if invoked from a container
# Otherwise, the default shared drive will be determined on the fly
class Input:
    __path = None

    def getPath():
        if Input.__path is None:
            Input()
        return Input.__path

    def __init__(self, path=None):
        if path:
            if Input.__path:
                raise Exception("Singleton Input cannot be constructed twice")
            else:
                Input.__path = path
        else:
            prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
            Input.__path = os.path.join(prefix, 'Shared drives/NIAAA_ASSIST/Data')
            