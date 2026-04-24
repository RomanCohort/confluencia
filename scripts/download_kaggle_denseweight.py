from kaggle.api.kaggle_api_extended import KaggleApi
import os, sys

DATASET = 'b0a096e3c550146f2a786f0ffd3c8bd37d68b04c7b09697efd282f91f8f6e36f'
OUTDIR = r'D:\IGEM集成方案\新建文件夹\DLEPS-main\DLEPS-main\data'

os.makedirs(OUTDIR, exist_ok=True)
api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    print('AUTH_ERROR', e)
    sys.exit(2)

print('Authenticated, starting download...')
api.dataset_download_files(DATASET, path=OUTDIR, unzip=True, quiet=False)
print('Download finished')
