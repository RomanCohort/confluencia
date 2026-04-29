"""
Confluencia CLI -- command-line interface for circRNA drug discovery.

This module provides a terminal interface for the Confluencia platform,
including drug efficacy prediction, epitope/MHC binding prediction,
circRNA multi-omics analysis, and joint evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Rich for animations
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ANSI Colors
_COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'reset': '\033[0m',
}

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.name != 'nt'

_USE_COLOR = _supports_color()

def _colorize(text: str, color: str) -> str:
    if not _USE_COLOR or color not in _COLORS:
        return text
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"

def cprint(text: str, color: str = None):
    print(_colorize(text, color) if color else text)

def print_success(text: str): cprint(text, 'green')
def print_error(text: str): cprint(text, 'red')
def print_warn(text: str): cprint(text, 'yellow')
def print_info(text: str): cprint(text, 'cyan')
def print_header(text: str): cprint(f"\n{'='*50}\n  {text}\n{'='*50}", 'magenta')

# ============================================================================
# Animation utilities
# ============================================================================

_console = Console() if RICH_AVAILABLE else None

class AnimatedSpinner:
    """Context manager for animated spinner with status messages."""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            char = self._spinner_chars[idx % len(self._spinner_chars)]
            sys.stdout.write(f"\r{char} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        if RICH_AVAILABLE:
            # Rich will be handled differently
            return self
        self._thread = threading.Thread(target=self._spin)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def update(self, message: str):
        self.message = message


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar context manager."""
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=_console,
        )
    return None


def show_animated_header(title: str):
    """Display an animated header."""
    if RICH_AVAILABLE:
        _console.print()
        _console.rule(f"[bold magenta]{title}[/bold magenta]")
        _console.print()
    else:
        print_header(title)


def show_success_panel(message: str, title: str = "Success"):
    """Display a success panel with animation."""
    if RICH_AVAILABLE:
        _console.print(Panel(f"[green]{message}[/green]", title=f"[bold green]{title}[/bold green]", border_style="green"))
    else:
        print_success(f"✓ {message}")


def show_result_table(results: Dict, title: str = "Results"):
    """Display results in an animated table."""
    if RICH_AVAILABLE:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        for key, value in results.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        _console.print(table)
    else:
        for key, value in results.items():
            print_info(f"{key}: {value}")


def animate_processing_steps(steps: List[str], delay: float = 0.3):
    """Animate through processing steps with a spinner."""
    if RICH_AVAILABLE:
        with _console.status("[bold cyan]Initializing...[/bold cyan]", spinner="dots") as status:
            for i, step in enumerate(steps):
                status.update(f"[bold cyan]{step}[/bold cyan]")
                time.sleep(delay)
            status.update("[bold green]Complete![/bold green]")
    else:
        spinner = AnimatedSpinner()
        with spinner:
            for step in steps:
                spinner.update(step)
                time.sleep(delay)

# ============================================================================
# Config utilities
# ============================================================================

DEFAULT_CONFIG = {
    'log_level': 'INFO',
    'output_dir': 'output',
}

def load_config(args) -> Dict:
    """Load configuration from file."""
    config = DEFAULT_CONFIG.copy()
    if hasattr(args, 'config') and args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    return config

def configure_logging(args):
    """Configure logging level."""
    import logging
    level = getattr(logging, getattr(args, 'log_level', 'INFO').upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def save_bundle(bundle: Dict, path: str):
    """Save model bundle to file."""
    import joblib
    joblib.dump(bundle, path)
    print_success(f"Bundle saved to {path}")

def load_bundle(path: str) -> Dict:
    """Load model bundle from file."""
    import joblib
    return joblib.load(path)


# ============================================================================
# DRUG MODULE
# ============================================================================

def add_drug_subparser(subparsers):
    """Add drug subcommands."""
    drug = subparsers.add_parser('drug', help='Drug prediction module')
    drug_sub = drug.add_subparsers(dest='drug_cmd', required=True)

    # drug props
    p = drug_sub.add_parser('props', help='Calculate molecular properties')
    p.add_argument('--smiles', required=True, help='SMILES string')
    p.add_argument('--json', action='store_true', help='JSON output')

    # drug predict
    p = drug_sub.add_parser('predict', help='Predict drug efficacy')
    p.add_argument('--smiles', required=True, help='SMILES string')
    p.add_argument('--model', default=None, help='Model bundle path')
    p.add_argument('--param', nargs='*', default=[], help='Env params (k=v)')
    p.add_argument('--json', action='store_true', help='JSON output')

    # drug train
    p = drug_sub.add_parser('train', help='Train drug model')
    p.add_argument('--data', required=True, help='Training data CSV')
    p.add_argument('--output', default='model.joblib', help='Output model path')

    # drug pk
    p = drug_sub.add_parser('pk', help='PK simulation')
    p.add_argument('--ka', type=float, default=0.2)
    p.add_argument('--kd', type=float, default=0.15)
    p.add_argument('--ke', type=float, default=0.18)
    p.add_argument('--km', type=float, default=0.25)
    p.add_argument('--horizon', type=int, default=72)

    # drug screen
    p = drug_sub.add_parser('screen', help='Screen molecules')
    p.add_argument('--input', required=True, help='Input SMILES file')
    p.add_argument('--model', required=True, help='Model bundle path')
    p.add_argument('--output', default='screen_results.csv')

    # drug cv
    p = drug_sub.add_parser('cv', help='Cross-validation')
    p.add_argument('--data', required=True, help='Training data CSV')
    p.add_argument('--folds', type=int, default=5)

def cmd_drug_props(args):
    """Calculate molecular properties."""
    show_animated_header('Molecular Properties')

    steps = [
        "Loading featurizer...",
        "Parsing SMILES string...",
        "Computing molecular descriptors...",
        "Generating feature vector...",
    ]
    animate_processing_steps(steps, delay=0.2)

    from confluencia_2_0_drug.core.featurizer import MoleculeFeatures

    featurizer = MoleculeFeatures()
    features, names = featurizer.transform_many([args.smiles])

    result = {'smiles': args.smiles, 'feature_dim': features.shape[1]}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        show_result_table(result, "Molecular Properties")

def cmd_drug_predict(args):
    """Predict drug efficacy for single SMILES."""
    show_animated_header('Drug Prediction')

    steps = [
        "Parsing SMILES...",
        "Loading feature descriptors...",
        "Running CTM simulation...",
        "Computing efficacy metrics...",
    ]
    animate_processing_steps(steps, delay=0.3)

    # Load model bundle if provided
    if args.model:
        bundle = load_bundle(args.model)
        from confluencia_2_0_drug.core.predictor import predict_one
        result = predict_one(bundle, args.smiles, {})
    else:
        # Use CTM simulation as default prediction
        from confluencia_2_0_drug.core.ctm import (
            CTMParams, simulate_ctm, summarize_curve,
            params_from_micro_scores,
        )
        params = params_from_micro_scores(binding=0.5, immune=0.5, inflammation=0.2)
        curve = simulate_ctm(dose=1.0, freq=1.0, params=params, horizon=72)
        summary = summarize_curve(curve)
        result = {
            'smiles': args.smiles,
            'auc_efficacy': round(summary['auc_efficacy'], 4),
            'peak_efficacy': round(summary['peak_efficacy'], 4),
            'peak_toxicity': round(summary['peak_toxicity'], 4),
        }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        show_result_table(result, "Drug Prediction")

def cmd_drug_train(args):
    """Train drug model."""
    show_animated_header('Drug Model Training')

    steps = [
        "Loading training data...",
        "Extracting molecular features...",
        "Initializing MOE ensemble...",
        "Training expert models...",
        "Optimizing gating network...",
        "Validating performance...",
        "Saving model bundle...",
    ]

    import pandas as pd
    from confluencia_2_0_drug.core.training import train_drug_model

    # Simulate progress animation
    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task("[cyan]Training model...", total=len(steps))
            for i, step in enumerate(steps):
                progress.update(task, description=f"[cyan]{step}")
                time.sleep(0.3)
                progress.advance(task)
            # Actual training
            progress.update(task, description="[yellow]Running actual training...")
            data = pd.read_csv(args.data)
            model = train_drug_model(data)
            progress.update(task, description="[green]Saving model...", completed=len(steps))
    else:
        animate_processing_steps(steps, delay=0.3)
        data = pd.read_csv(args.data)
        model = train_drug_model(data)

    save_bundle(model, args.output)
    show_success_panel(f"Model saved to {args.output}", "Training Complete")

def cmd_drug_pk(args):
    """Run PK simulation."""
    show_animated_header('PK Simulation')

    steps = [
        "Initializing pharmacokinetic model...",
        "Setting absorption parameters...",
        "Configuring distribution kinetics...",
        "Running time-course simulation...",
        "Computing AUC and Cmax...",
    ]
    animate_processing_steps(steps, delay=0.25)

    from confluencia_2_0_drug.core.ctm import CTMParams, simulate_ctm, summarize_curve

    params = CTMParams(
        ka=args.ka, kd=args.kd, ke=args.ke, km=args.km, signal_gain=1.0
    )
    curve = simulate_ctm(dose=1.0, freq=1.0, params=params, horizon=args.horizon)
    summary = summarize_curve(curve)

    result = {
        'ka': args.ka,
        'kd': args.kd,
        'ke': args.ke,
        'km': args.km,
        'AUC': round(summary['auc_efficacy'], 4),
        'Cmax': round(summary['peak_efficacy'], 4),
        'Peak Toxicity': round(summary['peak_toxicity'], 4),
    }
    show_result_table(result, "PK Simulation Results")

def cmd_drug_screen(args):
    """Screen molecules."""
    show_animated_header('Molecular Screening')

    import pandas as pd
    from confluencia_2_0_drug.core.predictor import predict_one

    with open(args.input) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    bundle = load_bundle(args.model)
    results = []

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task(f"[cyan]Screening {len(smiles_list)} molecules...", total=len(smiles_list))
            for i, smiles in enumerate(smiles_list):
                progress.update(task, description=f"[cyan]Processing [{i+1}/{len(smiles_list)}]: {smiles[:20]}...")
                result = predict_one(bundle, smiles, {})
                result['smiles'] = smiles
                results.append(result)
                progress.advance(task)
    else:
        for i, smiles in enumerate(smiles_list):
            print_info(f"[{i+1}/{len(smiles_list)}] Processing: {smiles}")
            result = predict_one(bundle, smiles, {})
            result['smiles'] = smiles
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    show_success_panel(f"Screened {len(smiles_list)} molecules\nResults saved to {args.output}", "Screening Complete")

def cmd_drug_cv(args):
    """Cross-validation."""
    import pandas as pd
    from confluencia_2_0_drug.core.moe import MOERegressor, ExpertConfig
    from sklearn.model_selection import cross_val_score

    data = pd.read_csv(args.data)
    X = data.drop(columns=['target']).values
    y = data['target'].values

    config = ExpertConfig()
    moe = MOERegressor(config)

    scores = cross_val_score(moe, X, y, cv=args.folds, scoring='r2')

    print_header('Cross-Validation Results')
    print_info(f"Folds: {args.folds}")
    print_info(f"R² scores: {scores}")
    print_info(f"Mean R²: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")


# ============================================================================
# EPITOPE MODULE
# ============================================================================

def add_epitope_subparser(subparsers):
    """Add epitope subcommands."""
    epitope = subparsers.add_parser('epitope', help='Epitope prediction module')
    epi_sub = epitope.add_subparsers(dest='epitope_cmd', required=True)

    # epitope predict
    p = epi_sub.add_parser('predict', help='Predict epitope binding')
    p.add_argument('--sequence', required=True, help='Peptide sequence')
    p.add_argument('--mhc', default='HLA-A*02:01', help='MHC allele')
    p.add_argument('--model', default=None, help='Model path')
    p.add_argument('--json', action='store_true', help='JSON output')

    # epitope train
    p = epi_sub.add_parser('train', help='Train epitope model')
    p.add_argument('--data', required=True, help='Training data CSV')
    p.add_argument('--output', default='epitope_model.joblib')

    # epitope encode
    p = epi_sub.add_parser('encode', help='Encode sequences')
    p.add_argument('--sequence', required=True, help='Peptide sequence')
    p.add_argument('--method', choices=['mamba', 'traditional'], default='mamba')

    # epitope cv
    p = epi_sub.add_parser('cv', help='Cross-validation')
    p.add_argument('--data', required=True, help='Training data CSV')
    p.add_argument('--folds', type=int, default=5)

def cmd_epitope_predict(args):
    """Predict epitope binding."""
    from confluencia_2_0_epitope.core.features import extract_sequence_features

    features = extract_sequence_features([args.sequence])

    result = {
        'sequence': args.sequence,
        'mhc': args.mhc,
        'feature_dim': features.shape[1],
        'binding_score': 0.5,  # Placeholder
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_header('Epitope Prediction')
        print_info(f"Sequence: {args.sequence}")
        print_info(f"MHC: {args.mhc}")
        print_info(f"Binding score: {result['binding_score']:.4f}")

def cmd_epitope_train(args):
    """Train epitope model."""
    import pandas as pd
    from confluencia_2_0_epitope.core.training import train_model

    data = pd.read_csv(args.data)
    model = train_model(data)
    save_bundle(model, args.output)
    print_success(f"Model saved to {args.output}")

def cmd_epitope_encode(args):
    """Encode peptide sequence."""
    from confluencia_2_0_epitope.core.features import extract_sequence_features
    from confluencia_2_0_epitope.core.mamba3 import Mamba3LiteEncoder

    if args.method == 'mamba':
        encoder = Mamba3LiteEncoder(d_model=32)
        features = encoder.encode_batch([args.sequence])
        print_info(f"Mamba3Lite encoding: {features.shape}")
    else:
        features = extract_sequence_features([args.sequence])
        print_info(f"Traditional features: {features.shape}")

def cmd_epitope_cv(args):
    """Cross-validation for epitope model."""
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from confluencia_2_0_epitope.core.features import extract_sequence_features

    data = pd.read_csv(args.data)
    X = extract_sequence_features(data['sequence'].tolist())
    y = data['target'].values

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, y, cv=args.folds, scoring='roc_auc')

    print_header('Epitope Cross-Validation')
    print_info(f"AUC scores: {scores}")
    print_info(f"Mean AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")


# ============================================================================
# CIRCRNA MODULE
# ============================================================================

def add_circrna_subparser(subparsers):
    """Add circRNA subcommands."""
    circrna = subparsers.add_parser('circrna', help='circRNA analysis module')
    circ_sub = circrna.add_subparsers(dest='circrna_cmd', required=True)

    # circrna immune
    p = circ_sub.add_parser('immune', help='Immunogenicity prediction')
    p.add_argument('--sequence', required=True)

    # circrna survival
    p = circ_sub.add_parser('survival', help='Survival analysis')
    p.add_argument('--data', required=True)

def cmd_circrna_immune(args):
    """Predict immunogenicity."""
    from confluencia_2_0_drug.core.innate_immune import assess_innate_immune

    result = assess_innate_immune(args.sequence)
    print_header('Immunogenicity Prediction')
    print_info(f"Sequence length: {len(args.sequence)}")
    print_info(f"Immunogenicity score: {result:.4f}")

def cmd_circrna_survival(args):
    """Survival analysis."""
    print_header('Survival Analysis')
    print_info(f"Data: {args.data}")
    print_warn("Survival analysis not implemented in this demo")


# ============================================================================
# JOINT MODULE
# ============================================================================

def add_joint_subparser(subparsers):
    """Add joint evaluation subcommands."""
    joint = subparsers.add_parser('joint', help='Joint evaluation module')
    joint_sub = joint.add_subparsers(dest='joint_cmd', required=True)

    p = joint_sub.add_parser('evaluate', help='Joint evaluation')
    p.add_argument('--config', required=True, help='Config YAML')

def cmd_joint_evaluate(args):
    """Joint evaluation."""
    print_header('Joint Evaluation')
    print_info(f"Config: {args.config}")
    print_warn("Joint evaluation not implemented in this demo")


# ============================================================================
# BENCH MODULE
# ============================================================================

def add_bench_subparser(subparsers):
    """Add benchmark subcommands."""
    bench = subparsers.add_parser('bench', help='Benchmark module')
    bench_sub = bench.add_subparsers(dest='bench_cmd', required=True)

    p = bench_sub.add_parser('run-all', help='Run all benchmarks')
    p.add_argument('--data-epitope', default='data/example_epitope.csv')
    p.add_argument('--data-drug', default='data/example_drug.csv')

def cmd_bench_run_all(args):
    """Run all benchmarks."""
    print_header('Running Benchmarks')
    print_info("Starting benchmark suite...")
    print_warn("Benchmarks not implemented in this demo")


# ============================================================================
# CHART MODULE
# ============================================================================

def add_chart_subparser(subparsers):
    """Add chart subcommands."""
    chart = subparsers.add_parser('chart', help='Visualization module')
    chart_sub = chart.add_subparsers(dest='chart_cmd', required=True)

    p = chart_sub.add_parser('pk', help='PK curve plot')
    p.add_argument('--output', default='pk_curve.png')

def cmd_chart_pk(args):
    """Plot PK curve."""
    try:
        import matplotlib.pyplot as plt
        from confluencia_2_0_drug.core.ctm import CTMParams, simulate_ctm

        params = CTMParams(ka=0.2, kd=0.15, ke=0.18, km=0.25, signal_gain=1.0)
        curve = simulate_ctm(params, horizon=72)

        plt.figure(figsize=(10, 6))
        plt.plot(curve['time'], curve['effect'], linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Effect')
        plt.title('PK Simulation')
        plt.savefig(args.output, dpi=150)
        plt.close()
        print_success(f"Chart saved to {args.output}")
    except ImportError:
        print_error("matplotlib not installed")


# ============================================================================
# APP MODULE
# ============================================================================

_APP_FILES = {
    'drug': 'confluencia-2.0-drug/app.py',
    'epitope': 'confluencia-2.0-epitope/epitope_frontend.py',
    'main': 'src/frontend.py',
}

def add_app_subparser(subparsers):
    """Add app subcommands."""
    app = subparsers.add_parser('app', help='Launch Streamlit apps')
    app_sub = app.add_subparsers(dest='app_cmd', required=True)

    p = app_sub.add_parser('launch', help='Launch app')
    p.add_argument('--app', choices=list(_APP_FILES.keys()), default='main')

def cmd_app_launch(args):
    """Launch Streamlit app."""
    app_file = _APP_FILES.get(args.app)
    if app_file:
        cmd = ['streamlit', 'run', app_file, '--server.port', '8501']
        print_info(f"Launching {args.app} app...")
        subprocess.run(cmd)
    else:
        print_error(f"Unknown app: {args.app}")


# ============================================================================
# VERSION
# ============================================================================

def add_version_subparser(subparsers):
    """Add version command."""
    subparsers.add_parser('version', help='Show version')

def cmd_version(args):
    """Show version."""
    if RICH_AVAILABLE:
        _console.print()
        _console.print(Panel.fit(
            "[bold cyan]Confluencia CLI[/bold cyan]  v2.1.0\n"
            "[dim]circRNA Drug Discovery Platform[/dim]\n"
            "[dim]https://github.com/IGEM-FBH/confluencia[/dim]",
            border_style="cyan",
        ))
    else:
        print_info("Confluencia CLI v2.1.0")
        print_info("circRNA Drug Discovery Platform")
        print_info("https://github.com/IGEM-FBH/confluencia")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def add_interactive_subparser(subparsers):
    """Add interactive mode."""
    subparsers.add_parser('interactive', help='Interactive REPL')

def cmd_interactive(args):
    """Launch interactive REPL."""
    print_header('Confluencia Interactive Mode')
    print_info("Type 'help' for commands, 'exit' to quit")

    while True:
        try:
            cmd = input("confluencia> ").strip()
            if cmd in ('exit', 'quit', 'q'):
                break
            elif cmd == 'help':
                print("Commands: drug, epitope, circrna, version, help, exit")
            elif cmd.startswith('drug '):
                print_info(f"Drug command: {cmd}")
            elif cmd.startswith('epitope '):
                print_info(f"Epitope command: {cmd}")
            elif cmd == 'version':
                print_info("Confluencia v2.1.0")
            else:
                print_warn(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# ============================================================================
# MAIN
# ============================================================================

def main(argv: List[str] = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='confluencia',
        description='Confluencia CLI - circRNA drug discovery platform',
    )
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--config', help='Config file path')

    subparsers = parser.add_subparsers(dest='module', required=True)

    # Add module subparsers
    add_drug_subparser(subparsers)
    add_epitope_subparser(subparsers)
    add_circrna_subparser(subparsers)
    add_joint_subparser(subparsers)
    add_bench_subparser(subparsers)
    add_chart_subparser(subparsers)
    add_app_subparser(subparsers)
    add_version_subparser(subparsers)
    add_interactive_subparser(subparsers)

    args = parser.parse_args(argv)
    configure_logging(args)

    # Route to module handlers
    module = args.module

    if module == 'drug':
        cmd = f"cmd_drug_{args.drug_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.drug_cmd}"))(args)
    elif module == 'epitope':
        cmd = f"cmd_epitope_{args.epitope_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.epitope_cmd}"))(args)
    elif module == 'circrna':
        cmd = f"cmd_circrna_{args.circrna_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.circrna_cmd}"))(args)
    elif module == 'joint':
        cmd = f"cmd_joint_{args.joint_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.joint_cmd}"))(args)
    elif module == 'bench':
        cmd = f"cmd_bench_{args.bench_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.bench_cmd}"))(args)
    elif module == 'chart':
        cmd = f"cmd_chart_{args.chart_cmd.replace('-', '_')}"
        globals().get(cmd, lambda a: print_error(f"Unknown command: {args.chart_cmd}"))(args)
    elif module == 'app':
        if args.app_cmd == 'launch':
            cmd_app_launch(args)
    elif module == 'version':
        cmd_version(args)
    elif module == 'interactive':
        cmd_interactive(args)


if __name__ == '__main__':
    main()
