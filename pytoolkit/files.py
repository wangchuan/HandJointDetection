import os

def dir(path, ext=None):
    if ext is '':
        ext = None
    files = [os.path.join(path, x) for x in os.listdir(path)]
    files = [x for x in files if os.path.isfile(x) and (ext==None or os.path.splitext(x)[-1]==ext)]
    return sorted(files)

def fileparts(filename):
    path, filename = os.path.split(filename)
    filename, ext = os.path.splitext(filename)
    return path, filename, ext

