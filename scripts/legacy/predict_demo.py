import os
import pandas as pd

from src.drug.torch_predictor import load_torch_bundle, predict_torch_batch, predict_torch_one

print('CWD', os.getcwd())
model_path = os.path.join('models','pretrained','test_drug_torch_pretrained.pt')
print('model_path exists:', os.path.exists(model_path))

b = load_torch_bundle(model_path)
print('bundle: featurizer_version=', b.featurizer_version, 'n_bits=', b.n_bits, 'input_dim=', b.input_dim)

# single prediction
smiles = 'CCO'
env = {c: float(b.env_medians.get(c, 0.0)) for c in b.env_cols}
# override some common cols if present
if 'dose' in b.env_cols:
    env['dose'] = 10.0
if 'freq' in b.env_cols:
    env['freq'] = 2.0
val = predict_torch_one(b, smiles=smiles, env_params=env)
print('single predict:', smiles, env, '->', val)

# batch predict first 5 rows
df = pd.read_csv(os.path.join('data', 'example_drug.csv'))
rows = df.head(5)
print('batch size', len(rows))
smiles_list = [str(r.get('smiles', '')) for _, r in rows.iterrows()]
env_list = [
    {c: (float(r[c]) if c in r and pd.notna(r[c]) else float(b.env_medians.get(c, 0.0))) for c in b.env_cols}
    for _, r in rows.iterrows()
]
preds = predict_torch_batch(b, smiles_list=smiles_list, env_params_list=env_list, batch_size=128)
for i, (s, env_params, p) in enumerate(zip(smiles_list, env_list, preds)):
    print(i, s, env_params, '->', float(p))
