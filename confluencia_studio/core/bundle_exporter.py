"""Reproducible Bundle Exporter for ConfluenciaStudio.

Creates self-contained ZIP archives of experiments including data,
code, models, and reproduction instructions.
"""

from __future__ import annotations

import json
import zipfile
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ReproducibleBundle:
    """Creates reproducible experiment bundles.

    Features:
    - Packages experiment data, code, and models
    - Generates reproduction instructions
    - Creates requirements.txt
    - Computes file checksums

    Usage:
        bundler = ReproducibleBundle()
        bundle_path = bundler.create_bundle(
            experiment_id="abc123",
            output_path="output/experiment_abc123.zip"
        )
    """

    def __init__(self, storage_dir: str = "~/.confluencia/experiments"):
        self.storage_dir = Path(storage_dir).expanduser()

    def create_bundle(
        self,
        experiment_id: str,
        output_path: str,
        include_data: bool = True,
        include_models: bool = True,
        include_code: bool = True,
        git_hash: Optional[str] = None,
    ) -> str:
        """Create a reproducible bundle for an experiment.

        Args:
            experiment_id: Experiment ID
            output_path: Output ZIP file path
            include_data: Include data files
            include_models: Include model files
            include_code: Include code files
            git_hash: Git commit hash for code reference

        Returns:
            Path to created bundle
        """
        # Load experiment data
        exp_data = self._load_experiment(experiment_id)
        if not exp_data:
            raise ValueError(f"Experiment {experiment_id} not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Experiment metadata
            zf.writestr("experiment.json", json.dumps(exp_data, indent=2, default=str))

            # 2. README with reproduction instructions
            readme = self._generate_readme(exp_data, git_hash)
            zf.writestr("README.md", readme)

            # 3. Requirements
            requirements = self._generate_requirements()
            zf.writestr("requirements.txt", requirements)

            # 4. Data files
            if include_data:
                self._add_data_files(zf, exp_data)

            # 5. Model files
            if include_models:
                self._add_model_files(zf, exp_data)

            # 6. Code reference
            if include_code:
                self._add_code_reference(zf, exp_data, git_hash)

            # 7. Checksums
            checksums = self._compute_checksums(zf)
            zf.writestr("checksums.json", json.dumps(checksums, indent=2))

        return str(output_path)

    def _load_experiment(self, exp_id: str) -> Optional[Dict]:
        """Load experiment data."""
        db_path = self.storage_dir / "experiments.json"
        if not db_path.exists():
            return None

        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get(exp_id)

    def _generate_readme(self, exp_data: Dict, git_hash: Optional[str] = None) -> str:
        """Generate README with reproduction instructions."""
        name = exp_data.get("name", "Experiment")
        module = exp_data.get("module", "unknown")
        params = exp_data.get("parameters", {})
        metrics = exp_data.get("metrics", {})

        lines = [
            f"# Reproducible Bundle: {name}\n",
            f"**Generated:** {datetime.now().isoformat()}\n",
            f"**Module:** {module}\n",
            f"**Status:** {exp_data.get('status', 'unknown')}\n",
            "---\n",
            "## Contents\n",
            "- `experiment.json` - Full experiment metadata",
            "- `requirements.txt` - Python dependencies",
            "- `data/` - Training and test data files",
            "- `models/` - Trained model files",
            "- `checksums.json` - File integrity checksums",
            "",
            "## Reproduce\n",
            "### 1. Setup Environment\n",
            "```bash",
            "python -m venv venv",
            "source venv/bin/activate  # Linux/Mac",
            "# or: venv\\Scripts\\activate  # Windows",
            "pip install -r requirements.txt",
            "```\n",
            "### 2. Reproduce Results\n",
        ]

        # Module-specific reproduction steps
        if module == "drug":
            data_path = params.get("data_path", "data/train.csv")
            model_type = params.get("model_type", "moe")
            lines.extend([
                "```bash",
                f"# Train model with original parameters",
                f"confluencia drug train --data {data_path} --model-type {model_type}",
                "```\n",
            ])
        elif module == "epitope":
            data_path = params.get("data_path", "data/epitope_train.csv")
            lines.extend([
                "```bash",
                f"confluencia epitope train --data {data_path} --use-mhc",
                "```\n",
            ])
        else:
            lines.extend([
                "```bash",
                "# Run with original parameters",
                f"python reproduce.py",
                "```\n",
            ])

        # Results
        if metrics:
            lines.append("## Original Results\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in sorted(metrics.items()):
                lines.append(f"| {key} | {value} |")
            lines.append("")

        # Git reference
        if git_hash:
            lines.append(f"## Code Reference\n")
            lines.append(f"Git commit: `{git_hash}`\n")

        return "\n".join(lines)

    def _generate_requirements(self) -> str:
        """Generate requirements.txt."""
        return """# Confluencia Core
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7

# Drug Module
rdkit>=2023.3
xgboost>=2.0
lightgbm>=4.0

# Epitope Module
biopython>=1.81

# circRNA Module
lifelines>=0.28

# Studio
plotly>=5.18
"""

    def _add_data_files(self, zf: zipfile.ZipFile, exp_data: Dict):
        """Add data files to bundle."""
        params = exp_data.get("parameters", {})
        artifacts = exp_data.get("artifacts", [])

        # Find data files from parameters
        data_paths = []
        for key in ["data_path", "data", "test_data", "train_data"]:
            if key in params:
                data_paths.append(params[key])

        # Add artifact data files
        for artifact in artifacts:
            if any(ext in artifact.lower() for ext in ['.csv', '.tsv', '.parquet', '.xlsx']):
                data_paths.append(artifact)

        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                arcname = f"data/{path.name}"
                try:
                    zf.write(str(path), arcname)
                except Exception:
                    pass

    def _add_model_files(self, zf: zipfile.ZipFile, exp_data: Dict):
        """Add model files to bundle."""
        artifacts = exp_data.get("artifacts", [])

        for artifact in artifacts:
            path = Path(artifact)
            if path.exists() and any(ext in path.suffix.lower() for ext in ['.pkl', '.joblib', '.pt', '.pth', '.h5', '.onnx']):
                arcname = f"models/{path.name}"
                try:
                    zf.write(str(path), arcname)
                except Exception:
                    pass

    def _add_code_reference(self, zf: zipfile.ZipFile, exp_data: Dict, git_hash: Optional[str] = None):
        """Add code reference or snapshot."""
        if git_hash:
            zf.writestr("code_reference.txt", f"Git commit: {git_hash}\n")
        else:
            # Create a simple reproduction script
            module = exp_data.get("module", "unknown")
            params = exp_data.get("parameters", {})

            script_lines = [
                '"""Auto-generated reproduction script."""',
                "import subprocess",
                "import sys",
                "",
                "# Install dependencies",
                "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])",
                "",
            ]

            if module == "drug":
                script_lines.extend([
                    "from confluencia_2_0_drug.core.pipeline import run_pipeline",
                    "",
                    "result = run_pipeline(",
                    f"    data_path='{params.get('data_path', 'data/train.csv')}',",
                    f"    model_type='{params.get('model_type', 'moe')}',",
                    f"    n_estimators={params.get('n_estimators', 100)},",
                    ")",
                    "print(f'R²: {result.r2:.4f}')",
                ])
            elif module == "epitope":
                script_lines.extend([
                    "from confluencia_2_0_epitope.core.pipeline import run_pipeline",
                    "",
                    "result = run_pipeline(",
                    f"    data_path='{params.get('data_path', 'data/epitope.csv')}',",
                    f"    use_mhc={params.get('use_mhc', True)},",
                    ")",
                    "print(f'AUC: {result.auc:.4f}')",
                ])
            else:
                script_lines.append("# TODO: Add module-specific reproduction code")

            zf.writestr("reproduce.py", "\n".join(script_lines))

    def _compute_checksums(self, zf: zipfile.ZipFile) -> Dict[str, str]:
        """Compute SHA256 checksums for all files in the bundle."""
        checksums = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info.filename) as f:
                sha256 = hashlib.sha256()
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
                checksums[info.filename] = sha256.hexdigest()
        return checksums
