import os

from huggingface_hub import hf_hub_download
from zipfile import ZipFile


def droidslam_download_weights():
    # Download droid.pth
    if not os.path.exists('droid.pth'):
        file_path = hf_hub_download(repo_id='vslamlab/droidslam', filename='droid.pth', repo_type='dataset',
                                    local_dir='.')
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('.')


if __name__ == "__main__":
    droidslam_download_weights()
