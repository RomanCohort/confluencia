import os
from src.pl_train import train

os.makedirs('tmp', exist_ok=True)
smiles = ['CCO','CC','CCC','CCN','CCOCC'] * 3
print('Starting quick PL training (epochs=3, use_ema=True)')
train(smiles, epochs=3, batch_size=4, out='tmp/pl_test.pth', use_cuda=False, use_ema=True, ema_decay=0.90)
print('Training finished')
print('tmp files:', os.listdir('tmp'))
