"""Enhanced REPL for Confluencia CLI.

Provides a feature-rich interactive terminal with:
- Command history with readline support
- Tab completion for modules and commands
- Variable storage and substitution
- LLM-assisted command suggestions
- Workspace persistence
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Windows VT100 support
_C = False
def _enable_windows_vt100():
    global _C
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            _C = True
        except Exception:
            pass

_enable_windows_vt100()

# Readline support
try:
    import readline
except ImportError:
    try:
        import pyreadline3 as readline  # type: ignore
    except ImportError:
        readline = None  # type: ignore

# Workspace paths
home = Path.home()
WORKSPACE_DIR = home / ".confluencia"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_FILE = WORKSPACE_DIR / "repl_workspace.json"
CONFIG_FILE = WORKSPACE_DIR / "repl_config.json"

# Module registry
MODULES = {
    "drug": [
        "train", "predict", "screen", "run", "run-predict", "cv",
        "suggest-env", "generate", "pk", "props", "fingerprint",
        "similarity", "pkpd", "train-torch", "predict-torch",
        "innate-immune", "reliability", "evaluate", "nca",
        "report", "evolve",
    ],
    "epitope": [
        "train", "predict", "screen", "cv", "sensitivity", "orf",
        "suggest-env", "evaluate", "reliability", "report",
        "esm2-encode", "esm2-batch", "encode", "mhc-encode",
        "bio", "acquire", "ckpt-list", "ckpt-load", "ckpt-cleanup",
        "moe-explain", "batch-orf", "fasta-crawl", "fasta-clean",
    ],
    "circrna": [
        "immune", "multiomics", "bulk", "tme", "survival",
        "pathway", "evasion", "cycle", "genomic", "fetch",
        "tmb", "cnv", "enrich", "survival-sign", "cross-val-survival",
        "stratify", "features", "train-survival", "cibersort-train",
    ],
    "joint": ["circrna-pk", "immune-abm", "tumor-killing", "evaluate", "batch"],
    "bench": ["run-all", "ablation", "baselines", "sensitivity", "clinical", "mamba", "stat-tests", "fetch-data", "quick"],
    "chart": ["pk", "regression", "importance", "compare", "sensitivity", "survival", "histogram", "scatter"],
    "app": ["launch"],
}

BUILTIN_COMMANDS = {
    "help": "Show help",
    "exit": "Exit REPL",
    "quit": "Exit REPL",
    "back": "Exit current module",
    "set": "Set variable (set key=value)",
    "unset": "Remove variable",
    "vars": "List variables",
    "ls": "List variables",
    "rm": "Delete variable",
    "clear": "Clear screen",
    "history": "Show command history",
    "version": "Show version",
    "time": "Time a command",
    "alias": "Create alias",
    "unalias": "Remove alias",
    "aliases": "List aliases",
    "save": "Save workspace",
    "load": "Load workspace",
    "save.image": "Save full workspace image",
    "getwd": "Print working directory",
    "setwd": "Set working directory",
    "cd": "Change directory",
    "dir": "List directory",
    "last": "Re-run last command",
    "options": "Set/display options",
    "demo": "Run demo",
    "llm": "Chat with LLM assistant",
}


class LLMController:
    """Simple LLM controller for REPL assistance."""

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"
        self._load_config()

    def _load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                self.api_key = cfg.get("api_key", self.api_key)
                self.base_url = cfg.get("base_url", self.base_url)
                self.model = cfg.get("model", self.model)
            except Exception:
                pass

    def _save_config(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({"api_key": self.api_key, "base_url": self.base_url, "model": self.model}, f, indent=2)

    def chat(self, message: str) -> str:
        """Send message to LLM and return response."""
        if not self.api_key:
            return "Error: API key not set. Use: llm set-key <key>"

        import urllib.request
        import urllib.error

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an AI assistant for Confluencia, a circRNA drug discovery platform. Be concise."},
                {"role": "user", "content": message},
            ],
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"


class EnhancedREPL:
    """Enhanced interactive REPL for Confluencia.

    Features:
    - Module-based command navigation
    - Tab completion
    - Variable storage with $substitution
    - Command history with readline
    - LLM-assisted help
    - Workspace persistence
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._module: Optional[str] = None
        self._variables: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}
        self._history: List[str] = []
        self._last_command: Optional[str] = None
        self._llm = LLMController()
        self._running = True

        self._load_workspace()
        self._setup_readline()
        self._load_config()

    def _load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                self._aliases = cfg.get("aliases", {})
            except Exception:
                pass

    def _save_config(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({"aliases": self._aliases}, f, indent=2)

    def _load_workspace(self):
        if WORKSPACE_FILE.exists():
            try:
                with open(WORKSPACE_FILE) as f:
                    data = json.load(f)
                self._variables = data.get("variables", {})
                self._history = data.get("history", [])
            except Exception:
                pass

    def _save_workspace(self):
        WORKSPACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKSPACE_FILE, "w") as f:
            json.dump({"variables": self._variables, "history": self._history[-500:]}, f, indent=2)

    def _setup_readline(self):
        if readline is None:
            return

        history_file = WORKSPACE_DIR / "repl_history"
        try:
            readline.read_history_file(str(history_file))
        except FileNotFoundError:
            pass

        readline.set_completer(self._completer)
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")

        # Save history on exit
        try:
            readline.write_history_file(str(history_file))
        except Exception:
            pass

    def _save_history(self):
        if readline is None:
            return
        history_file = WORKSPACE_DIR / "repl_history"
        try:
            readline.write_history_file(str(history_file))
        except Exception:
            pass

    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for modules and commands."""
        options = []

        # If we're in a module, complete its commands
        if self._module and self._module in MODULES:
            options = [f"{self._module} {cmd}" for cmd in MODULES[self._module] if cmd.startswith(text)]
            if not options:
                options = [cmd for cmd in MODULES[self._module] if cmd.startswith(text)]
        else:
            # Complete module names and builtins
            options = [m for m in list(MODULES.keys()) + list(BUILTIN_COMMANDS.keys()) if m.startswith(text)]

        if state < len(options):
            return options[state]
        return None

    def _substitute_variables(self, line: str) -> str:
        """Replace $variable references with their values."""
        def _replace(match):
            name = match.group(1)
            value = self._variables.get(name)
            if value is not None:
                return str(value)
            return match.group(0)

        return re.sub(r'\$(\w+)', _replace, line)

    def _prompt(self) -> str:
        """Generate prompt string."""
        if self._module:
            return f"\033[94mconfluencia:{self._module}\033[0m> "
        return "\033[94mconfluencia\033[0m> "

    def _print_banner(self):
        """Print welcome banner."""
        print(f"""
\033[95m{'='*55}
  Confluencia v2.1.0 — Interactive REPL
  circRNA Drug Discovery Platform
{'='*55}\033[0m

Type \033[93mhelp\033[0m for commands, \033[93m<module>\033[0m to enter a module, \033[93mexit\033[0m to quit.
Modules: \033[92m{' '.join(MODULES.keys())}\033[0m
""")

    def _dispatch(self, line: str) -> bool:
        """Dispatch a command line. Returns False to exit."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True

        # Substitute variables
        line = self._substitute_variables(line)

        # Check aliases
        parts = line.split()
        if parts[0] in self._aliases:
            line = self._aliases[parts[0]] + " " + " ".join(parts[1:])
            parts = line.split()

        cmd = parts[0].lower()
        args = parts[1:]

        # Built-in commands
        if cmd == "exit" or cmd == "quit":
            return self._cmd_exit(args)
        if cmd == "help":
            return self._cmd_help(args)
        if cmd == "back":
            return self._cmd_back(args)
        if cmd == "set":
            return self._cmd_set(args)
        if cmd == "unset":
            return self._cmd_unset(args)
        if cmd in ("vars", "ls"):
            return self._cmd_vars(args)
        if cmd == "rm":
            return self._cmd_rm(args)
        if cmd == "clear":
            return self._cmd_clear(args)
        if cmd == "history":
            return self._cmd_history(args)
        if cmd == "version":
            return self._cmd_version(args)
        if cmd == "time":
            return self._cmd_time(args)
        if cmd == "alias":
            return self._cmd_alias(args)
        if cmd == "unalias":
            return self._cmd_unalias(args)
        if cmd == "aliases":
            return self._cmd_aliases(args)
        if cmd == "save":
            return self._cmd_save(args)
        if cmd == "load":
            return self._cmd_load(args)
        if cmd in ("save.image", "save_image"):
            return self._cmd_save_image(args)
        if cmd in ("getwd", "pwd"):
            return self._cmd_getwd(args)
        if cmd in ("setwd", "cd"):
            return self._cmd_cd(args)
        if cmd == "dir" or cmd == "ls" and not args:
            return self._cmd_dir(args)
        if cmd == "last":
            return self._cmd_last(args)
        if cmd == "options":
            return self._cmd_options(args)
        if cmd == "demo":
            return self._cmd_demo(args)
        if cmd == "llm":
            return self._cmd_llm(args)

        # Module entry
        if cmd in MODULES and not self._module:
            self._module = cmd
            print(f"\033[92m  Entered module: {cmd}\033[0m")
            return True

        # Pipeline command
        return self._execute_pipeline_command(cmd, args)

    def _execute_pipeline_command(self, cmd: str, args_list: List[str]) -> bool:
        """Execute a pipeline command via CLI."""
        from confluencia_cli.cli import main as cli_main

        if self._module:
            full_cmd = [self._module, cmd] + args_list
        else:
            full_cmd = [cmd] + args_list

        start = time.time()
        try:
            cli_main(full_cmd)
        except SystemExit:
            pass
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            traceback.print_exc()

        elapsed = time.time() - start
        print(f"\033[90m  [{elapsed:.2f}s]\033[0m")
        return True

    # --- Built-in command handlers ---

    def _cmd_help(self, args):
        if args and args[0] in MODULES:
            print(f"\n\033[93m{args[0]} commands:\033[0m")
            for cmd in MODULES[args[0]]:
                print(f"  {cmd}")
        elif args and args[0] in BUILTIN_COMMANDS:
            print(f"  {args[0]}: {BUILTIN_COMMANDS[args[0]]}")
        else:
            print(f"""
\033[93mBuilt-in commands:\033[0m""")
            for name, desc in BUILTIN_COMMANDS.items():
                print(f"  \033[92m{name:15s}\033[0m {desc}")
            print(f"""
\033[93mModules:\033[0m""")
            for mod, cmds in MODULES.items():
                print(f"  \033[92m{mod:15s}\033[0m {len(cmds)} commands")
            print("""
\033[90mType <module> to enter, <module> <command> [args] to execute.\033[0m""")
        return True

    def _cmd_exit(self, args):
        self._save_workspace()
        self._save_history()
        print("\033[93mGoodbye!\033[0m")
        return False

    def _cmd_back(self, args):
        if self._module:
            print(f"\033[93m  Left module: {self._module}\033[0m")
            self._module = None
        else:
            print("  Already at top level")
        return True

    def _cmd_set(self, args):
        if not args:
            for k, v in self._variables.items():
                print(f"  {k} = {v}")
            return True
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1)
                self._variables[k] = v
                print(f"  {k} = {v}")
        self._save_workspace()
        return True

    def _cmd_unset(self, args):
        for k in args:
            if k in self._variables:
                del self._variables[k]
                print(f"  Unset: {k}")
        self._save_workspace()
        return True

    def _cmd_vars(self, args):
        if not self._variables:
            print("  No variables set")
            return True
        for k, v in self._variables.items():
            print(f"  {k} = {v}")
        return True

    def _cmd_rm(self, args):
        return self._cmd_unset(args)

    def _cmd_clear(self, args):
        os.system("cls" if sys.platform == "win32" else "clear")
        return True

    def _cmd_history(self, args):
        n = int(args[0]) if args else 20
        for i, h in enumerate(self._history[-n:]):
            print(f"  {i+1:4d}  {h}")
        return True

    def _cmd_version(self, args):
        print("  Confluencia v2.1.0")
        return True

    def _cmd_time(self, args):
        if not args:
            print("  Usage: time <command>")
            return True
        start = time.time()
        self._dispatch(" ".join(args))
        elapsed = time.time() - start
        print(f"\033[93m  Elapsed: {elapsed:.3f}s\033[0m")
        return True

    def _cmd_alias(self, args):
        if len(args) < 2:
            print("  Usage: alias <name> <command>")
            return True
        name, *rest = args
        self._aliases[name] = " ".join(rest)
        self._save_config()
        print(f"  alias {name} = {' '.join(rest)}")
        return True

    def _cmd_unalias(self, args):
        for name in args:
            if name in self._aliases:
                del self._aliases[name]
        self._save_config()
        return True

    def _cmd_aliases(self, args):
        if not self._aliases:
            print("  No aliases set")
            return True
        for k, v in self._aliases.items():
            print(f"  {k} = {v}")
        return True

    def _cmd_save(self, args):
        self._save_workspace()
        print(f"  Workspace saved to {WORKSPACE_FILE}")
        return True

    def _cmd_load(self, args):
        path = Path(args[0]) if args else WORKSPACE_FILE
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._variables = data.get("variables", {})
            print(f"  Loaded {len(self._variables)} variables from {path}")
        else:
            print(f"  File not found: {path}")
        return True

    def _cmd_save_image(self, args):
        self._save_workspace()
        print(f"  Workspace image saved")
        return True

    def _cmd_getwd(self, args):
        print(f"  {os.getcwd()}")
        return True

    def _cmd_cd(self, args):
        if args:
            try:
                os.chdir(args[0])
                print(f"  {os.getcwd()}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  {os.getcwd()}")
        return True

    def _cmd_dir(self, args):
        path = Path(args[0]) if args else Path(".")
        for p in sorted(path.iterdir()):
            marker = "/" if p.is_dir() else ""
            print(f"  {p.name}{marker}")
        return True

    def _cmd_last(self, args):
        if self._last_command:
            print(f"  Re-running: {self._last_command}")
            return self._dispatch(self._last_command)
        print("  No last command")
        return True

    def _cmd_options(self, args):
        if not args:
            print(f"  module = {self._module}")
            print(f"  llm.model = {self._llm.model}")
            print(f"  llm.base_url = {self._llm.base_url}")
            return True
        if len(args) >= 2 and args[0] == "llm.model":
            self._llm.model = args[1]
            self._llm._save_config()
            print(f"  LLM model set to {args[1]}")
        return True

    def _cmd_demo(self, args):
        print("\033[93m  Running Confluencia demo...\033[0m")
        self._dispatch("drug predict --smiles CCO")
        self._dispatch("drug pk --ka 0.25")
        self._dispatch("version")
        return True

    def _cmd_llm(self, args):
        if not args:
            print("  Usage: llm <message> | llm set-key <key> | llm model <name>")
            return True

        if args[0] == "set-key":
            if len(args) < 2:
                print("  Usage: llm set-key <api-key>")
                return True
            self._llm.api_key = args[1]
            self._llm._save_config()
            print("  API key saved")
            return True

        if args[0] == "model":
            if len(args) < 2:
                print(f"  Current model: {self._llm.model}")
                return True
            self._llm.model = args[1]
            self._llm._save_config()
            print(f"  Model set to {args[1]}")
            return True

        message = " ".join(args)
        print(f"\033[93m  Asking LLM...\033[0m")
        response = self._llm.chat(message)
        print(f"\n{response}\n")
        return True

    def run(self):
        """Main REPL loop."""
        self._print_banner()

        while self._running:
            try:
                prompt = self._prompt()
                line = input(prompt).strip()
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print("\n\033[93mGoodbye!\033[0m")
                break

            if not line:
                continue

            self._history.append(line)
            self._last_command = line

            should_continue = self._dispatch(line)
            if not should_continue:
                break

        self._save_workspace()
        self._save_history()
