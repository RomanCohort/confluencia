"""Function Registry for LLM Tool Calling.

Defines Confluencia commands as callable tools that can be used by LLMs
via function calling / tool use APIs (OpenAI, DeepSeek, Anthropic).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""
    name: str
    description: str
    parameters: Dict  # JSON Schema
    handler: Optional[Callable] = None


# JSON Schema type definitions
STRING_TYPE = {"type": "string"}
INTEGER_TYPE = {"type": "integer"}
NUMBER_TYPE = {"type": "number"}
BOOLEAN_TYPE = {"type": "boolean"}

def enum_type(values: List[str]) -> Dict:
    return {"type": "string", "enum": values}


# Core Confluencia Tool Definitions
CONFLUENCIA_TOOLS: List[Dict] = [
    # ==================== DRUG MODULE ====================
    {
        "type": "function",
        "function": {
            "name": "drug_train",
            "description": "Train a drug efficacy prediction model. Use this when the user wants to train a new model on drug data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to training data CSV file (required)"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["rf", "xgb", "lgbm", "mlp", "moe"],
                        "description": "Model type: rf (Random Forest), xgb (XGBoost), lgbm (LightGBM), mlp (Neural Network), moe (Mixture of Experts, default)"
                    },
                    "n_estimators": {
                        "type": "integer",
                        "description": "Number of trees for RF/XGB/LGBM (default: 100)"
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Test set proportion (default: 0.2)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output model path"
                    },
                    "use_cross_features": {
                        "type": "boolean",
                        "description": "Enable cross features for better generalization (recommended)"
                    }
                },
                "required": ["data_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drug_predict",
            "description": "Predict drug efficacy for molecules. Use when user wants to predict efficacy for one or more SMILES.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Single SMILES string to predict"
                    },
                    "data_path": {
                        "type": "string",
                        "description": "CSV file with SMILES column for batch prediction"
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Path to trained model (required)"
                    }
                },
                "required": ["model_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drug_screen",
            "description": "Screen molecules for drug efficacy and rank by predicted score. Use for virtual screening.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "CSV file with SMILES to screen"
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Trained model path"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top candidates to return (default: 10)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum efficacy threshold filter"
                    }
                },
                "required": ["data_path", "model_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drug_pk",
            "description": "Simulate pharmacokinetics (PK) for a drug. Generates concentration-time curves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ka": {
                        "type": "number",
                        "description": "Absorption rate constant (1/h)"
                    },
                    "kd": {
                        "type": "number",
                        "description": "Distribution rate constant (1/h)"
                    },
                    "ke": {
                        "type": "number",
                        "description": "Elimination rate constant (1/h)"
                    },
                    "dose": {
                        "type": "number",
                        "description": "Dose in mg"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Simulation duration in hours"
                    }
                },
                "required": ["ka", "kd", "ke"]
            }
        }
    },

    # ==================== EPITOPE MODULE ====================
    {
        "type": "function",
        "function": {
            "name": "epitope_predict",
            "description": "Predict epitope-MHC binding affinity. Use for vaccine design and immunogenicity prediction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "epitope": {
                        "type": "string",
                        "description": "Peptide sequence (8-11 amino acids)"
                    },
                    "allele": {
                        "type": "string",
                        "description": "MHC allele (e.g., HLA-A*02:01)"
                    },
                    "data_path": {
                        "type": "string",
                        "description": "CSV with epitope sequences for batch prediction"
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Trained model path"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "epitope_train",
            "description": "Train epitope-MHC binding prediction model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "IEDB-format training data"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["rf", "xgb", "mlp", "moe", "mamba"],
                        "description": "Model architecture"
                    },
                    "use_mhc": {
                        "type": "boolean",
                        "description": "Include MHC pseudo-sequence features (recommended)"
                    }
                },
                "required": ["data_path"]
            }
        }
    },

    # ==================== CIRCRNA MODULE ====================
    {
        "type": "function",
        "function": {
            "name": "circrna_immune",
            "description": "Analyze circRNA immunogenicity. Predict immune response to circRNA therapeutics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "circRNA expression data"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for results"
                    }
                },
                "required": ["data_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "circrna_survival",
            "description": "Perform survival analysis on circRNA expression data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Expression data with survival info"
                    },
                    "circrna": {
                        "type": "string",
                        "description": "circRNA ID to analyze"
                    }
                },
                "required": ["data_path"]
            }
        }
    },

    # ==================== JOINT MODULE ====================
    {
        "type": "function",
        "function": {
            "name": "joint_evaluate",
            "description": "Joint drug-epitope efficacy evaluation. Combines drug efficacy, epitope binding, and PK simulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Drug SMILES string"
                    },
                    "epitope": {
                        "type": "string",
                        "description": "Epitope peptide sequence"
                    },
                    "allele": {
                        "type": "string",
                        "description": "MHC allele"
                    },
                    "dose": {
                        "type": "number",
                        "description": "Drug dose in mg"
                    },
                    "frequency": {
                        "type": "number",
                        "description": "Dosing frequency per day"
                    }
                },
                "required": ["smiles", "epitope", "allele"]
            }
        }
    },

    # ==================== CHART MODULE ====================
    {
        "type": "function",
        "function": {
            "name": "chart_pk",
            "description": "Generate PK concentration-time curve visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "PK simulation data"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output plot path"
                    }
                },
                "required": ["data_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chart_regression",
            "description": "Generate regression diagnostic plots (predicted vs actual, residuals).",
            "parameters": {
                "type": "object",
                "properties": {
                    "y_true_path": {
                        "type": "string",
                        "description": "True values file or column name"
                    },
                    "y_pred_path": {
                        "type": "string",
                        "description": "Predicted values file or column name"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output plot path"
                    }
                },
                "required": []
            }
        }
    },

    # ==================== EXPERIMENT MANAGEMENT ====================
    {
        "type": "function",
        "function": {
            "name": "experiment_list",
            "description": "List all tracked experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "enum": ["drug", "epitope", "circrna", "joint", "all"],
                        "description": "Filter by module"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "all"],
                        "description": "Filter by status"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "model_list",
            "description": "List all registered models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_filter": {
                        "type": "string",
                        "description": "Filter models by name pattern"
                    }
                },
                "required": []
            }
        }
    },

    # ==================== PYTHON EXECUTION ====================
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python code. Use for data manipulation, custom analysis, or visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents. Use to inspect data files or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to read (default: all)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., *.csv)"
                    }
                },
                "required": ["directory"]
            }
        }
    },
]


def get_tool_definitions() -> List[Dict]:
    """Get all tool definitions for LLM function calling."""
    return CONFLUENCIA_TOOLS


def get_tool_by_name(name: str) -> Optional[Dict]:
    """Get a specific tool definition by name."""
    for tool in CONFLUENCIA_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def execute_tool_call(tool_name: str, arguments: Dict, workspace: Optional[Dict] = None) -> str:
    """Execute a tool call and return the result.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        workspace: Optional workspace dict with variables

    Returns:
        Result string from tool execution
    """
    # Map tool names to Confluencia CLI commands
    cli_mapping = {
        "drug_train": _execute_drug_train,
        "drug_predict": _execute_drug_predict,
        "drug_screen": _execute_drug_screen,
        "drug_pk": _execute_drug_pk,
        "epitope_predict": _execute_epitope_predict,
        "epitope_train": _execute_epitope_train,
        "circrna_immune": _execute_circrna_immune,
        "circrna_survival": _execute_circrna_survival,
        "joint_evaluate": _execute_joint_evaluate,
        "chart_pk": _execute_chart_pk,
        "chart_regression": _execute_chart_regression,
        "experiment_list": _execute_experiment_list,
        "model_list": _execute_model_list,
        "python_exec": _execute_python,
        "read_file": _execute_read_file,
        "list_files": _execute_list_files,
    }

    handler = cli_mapping.get(tool_name)
    if not handler:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        return handler(arguments, workspace)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# ==================== TOOL HANDLERS ====================

def _run_cli_command(module: str, command: str, args: Dict) -> str:
    """Run a Confluencia CLI command and capture output."""
    cmd_parts = [sys.executable, "-m", "confluencia_cli", module, command]

    for key, value in args.items():
        if value is None:
            continue
        key_arg = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key_arg}")
        elif isinstance(value, (list, dict)):
            cmd_parts.extend([f"--{key_arg}", json.dumps(value)])
        else:
            cmd_parts.extend([f"--{key_arg}", str(value)])

    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(Path.cwd())
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 600 seconds"
    except FileNotFoundError:
        return f"Error: Could not find confluencia_cli. Make sure it's installed."
    except Exception as e:
        return f"Error: {str(e)}"


def _execute_drug_train(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("drug", "train", args)


def _execute_drug_predict(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("drug", "predict", args)


def _execute_drug_screen(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("drug", "screen", args)


def _execute_drug_pk(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("drug", "pk", args)


def _execute_epitope_predict(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("epitope", "predict", args)


def _execute_epitope_train(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("epitope", "train", args)


def _execute_circrna_immune(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("circrna", "immune", args)


def _execute_circrna_survival(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("circrna", "survival", args)


def _execute_joint_evaluate(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("joint", "evaluate", args)


def _execute_chart_pk(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("chart", "pk", args)


def _execute_chart_regression(args: Dict, workspace: Optional[Dict]) -> str:
    return _run_cli_command("chart", "regression", args)


def _execute_experiment_list(args: Dict, workspace: Optional[Dict]) -> str:
    """List experiments from the experiment tracker."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()

        module_filter = args.get("module", "all")
        status_filter = args.get("status", "all")

        experiments = tracker.list_experiments(
            module=module_filter if module_filter != "all" else None,
            status=status_filter if status_filter != "all" else None
        )

        if not experiments:
            return "No experiments found."

        lines = ["# Experiments\n"]
        for exp in experiments:
            lines.append(f"- **{exp.name}** ({exp.id})")
            lines.append(f"  - Module: {exp.module}")
            lines.append(f"  - Status: {exp.status}")
            lines.append(f"  - Created: {exp.created_at}")
            if exp.metrics:
                lines.append(f"  - Metrics: {json.dumps(exp.metrics)}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        return "Experiment tracker not available."


def _execute_model_list(args: Dict, workspace: Optional[Dict]) -> str:
    """List models from the model registry."""
    try:
        from confluencia_studio.core.model_registry import get_model_registry
        registry = get_model_registry()

        name_filter = args.get("name_filter")
        models = registry.list_models(name_filter)

        if not models:
            return "No models found."

        lines = ["# Models\n"]
        for model in models:
            lines.append(f"- **{model.name}** v{model.version} ({model.id})")
            lines.append(f"  - Type: {model.model_type}")
            lines.append(f"  - Path: {model.path}")
            lines.append(f"  - Created: {model.created_at}")
            if model.metrics:
                lines.append(f"  - Metrics: {json.dumps(model.metrics)}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        return "Model registry not available."


def _execute_python(args: Dict, workspace: Optional[Dict]) -> str:
    """Execute Python code."""
    code = args.get("code", "")
    if not code:
        return "Error: No code provided."

    import io
    import sys

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    # Create execution namespace
    namespace = dict(workspace) if workspace else {}
    namespace.update({
        '__builtins__': __builtins__,
        'pd': __import__('pandas') if 'pandas' in sys.modules else None,
        'np': __import__('numpy') if 'numpy' in sys.modules else None,
        'plt': __import__('matplotlib.pyplot') if 'matplotlib' in sys.modules else None,
    })

    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
        error = sys.stderr.getvalue()

        result = output
        if error:
            result += f"\n[stderr]\n{error}"

        return result or "(no output)"

    except Exception as e:
        import traceback
        return f"Error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _execute_read_file(args: Dict, workspace: Optional[Dict]) -> str:
    """Read a file."""
    path = args.get("path")
    if not path:
        return "Error: No path provided."

    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"

        lines_limit = args.get("lines")

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            if lines_limit:
                lines = [next(f) for _ in range(lines_limit)]
                content = ''.join(lines)
                content += f"\n... (showing first {lines_limit} lines)"
            else:
                content = f.read()

        return content

    except Exception as e:
        return f"Error reading file: {str(e)}"


def _execute_list_files(args: Dict, workspace: Optional[Dict]) -> str:
    """List files in a directory."""
    directory = args.get("directory")
    if not directory:
        return "Error: No directory provided."

    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {directory}"

        pattern = args.get("pattern", "*")
        files = list(dir_path.glob(pattern))

        lines = [f"# Files in {directory}\n"]
        for f in sorted(files):
            if f.is_dir():
                lines.append(f"📁 {f.name}/")
            else:
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                lines.append(f"📄 {f.name} ({size_str})")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing files: {str(e)}"
