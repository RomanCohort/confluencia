# ED2Mol Integration Notes

This project integrates molecule generation from:
- https://github.com/pineappleK/ED2Mol

## Minimal setup

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_ed2mol.ps1
```

Then create/configure the ED2Mol environment and model weights according to ED2Mol README.

## Runtime in Confluencia

In the evolution panel, provide:
- `ED2Mol repo dir`: local folder containing `Generate.py`
- `ED2Mol config path`: an ED2Mol YAML config that can run generation
- `ED2Mol python cmd`: python executable in ED2Mol environment

Confluencia patches only `output_dir` in a temp config and runs:

```bash
python Generate.py --config <temp_config>
```

Generated molecules are parsed recursively from output directory (`.csv/.smi/.txt/.sdf`).
