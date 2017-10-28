"""Downloads Udacity German Signs
"""

from urllib.request import urlretrieve
import os
from tqdm import tqdm
import zipfile


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_and_extract(url, path_to_file, path_to_extract):
    if not os.path.isfile(path_to_file):
        # Download
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Dataset') as pbar:
            urlretrieve(url, path_to_file, pbar.hook)

    if not os.path.isdir(path_to_extract):
        os.makedirs(path_to_extract)
        # extract
        with zipfile.ZipFile(path_to_file, 'r') as zf:
            zf.extractall(path_to_extract)
            zf.close()
