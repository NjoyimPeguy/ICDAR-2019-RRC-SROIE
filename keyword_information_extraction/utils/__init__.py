import os.path as osp


def get_basename(file):
    return osp.splitext(osp.basename(file))[0]
