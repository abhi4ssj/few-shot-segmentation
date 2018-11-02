import os


def create_if_not(path):
    if not os.path.exists(path):
        os.makedirs(path)
