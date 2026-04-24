#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Universal PyInstaller Runtime Setup - GUI Version

A generic tool to check and install VC++ Redistributables for any PyInstaller app.

Usage:
    python run_setup.py                        # GUI mode
    python run_setup.py --cli                  # CLI mode
    python run_setup.py --check                # Check only, no install
    python run_setup.py --silent               # Silent install
    python run_setup.py --exe "myapp.exe"      # Specify target exe

Pack as exe:
    pyinstaller --onefile --windowed --name "RuntimeSetup" run_setup.py
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# tkinter availability check
try:
    import tkinter as tk
    from tkinter import messagebox, ttk, scrolledtext
    HAS_TK = True
except ImportError:
    HAS_TK = False

# =============================================================================
# Constants
# =============================================================================

APP_NAME = "Runtime Setup"
VERSION = "1.0.0"

# Microsoft download URLs
VC2010_X86_URL = "https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe"
VC2010_X64_URL = "https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe"
VC2015_X86_URL = "https://aka.ms/vs/17/release/vc_redist.x86.exe"
VC2015_X64_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

# Registry paths
VC_REGS = {
    "vc2010_x86": (r"SOFTWARE\Microsoft\VisualStudio\10.0\VC\VCRedist\x86", "Installed"),
    "vc2010_x64": (r"SOFTWARE\Microsoft\VisualStudio\10.0\VC\VCRedist\x64", "Installed"),
    "vc2015_x86": (r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86", "Installed"),
    "vc2015_x64": (r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64", "Installed"),
}
VC_WOW64_REGS = {
    "vc2010_x86": r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\10.0\VC\VCRedist\x86",
    "vc2010_x64": r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\10.0\VC\VCRedist\x64",
    "vc2015_x86": r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86",
    "vc2015_x64": r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
}


# =============================================================================
# Registry Functions
# =============================================================================

def _query_reg(key_path: str, value_name: str = "Installed") -> Optional[int]:
    """Query registry, return value or None."""
    if sys.platform != "win32":
        return None
    import winreg
    for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        try:
            key = winreg.OpenKey(root, key_path, 0, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
            val, _ = winreg.QueryValueEx(key, value_name)
            winreg.CloseKey(key)
            return val
        except (FileNotFoundError, OSError):
            continue
    return None


def check_vc(name: str) -> Tuple[bool, str]:
    """Check if VC runtime is installed."""
    if sys.platform != "win32":
        return False, "Non-Windows"

    primary = VC_REGS.get(name)
    wow64 = VC_WOW64_REGS.get(name)
    if not primary:
        return False, "Unknown"

    val = _query_reg(primary[0], primary[1])
    if val == 1:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, primary[0], 0, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
            ver, _ = winreg.QueryValueEx(key, "Version")
            winreg.CloseKey(key)
            return True, f"v{ver}"
        except Exception:
            return True, "Installed"

    if wow64:
        val = _query_reg(wow64, "Installed")
        if val == 1:
            return True, "Installed"

    return False, "Not installed"


def check_all() -> Dict[str, Tuple[bool, str]]:
    return {name: check_vc(name) for name in VC_REGS}


# =============================================================================
# Download & Install
# =============================================================================

def download(url: str, dest: Path, progress=None) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress and total > 0:
                    progress(downloaded, total, int(downloaded * 100 / total))
        return True
    except Exception:
        return False


def install(installer: Path, silent: bool = True) -> Tuple[bool, str]:
    if not installer.exists():
        return False, "Installer not found"
    try:
        args = [str(installer)]
        if silent:
            args.extend(["/quiet", "/norestart"])
        result = subprocess.run(args, capture_output=True, timeout=600)
        if result.returncode == 0:
            return True, "Installed"
        elif result.returncode == 3010:
            return True, "Installed (restart required)"
        return False, f"Failed (code {result.returncode})"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Main Logic
# =============================================================================

RUNTIME_DEFS = [
    ("vc2015_x64", "VC++ 2015-2022 (x64)", VC2015_X64_URL),
    ("vc2015_x86", "VC++ 2015-2022 (x86)", VC2015_X86_URL),
    ("vc2010_x64", "VC++ 2010 SP1 (x64)", VC2010_X64_URL),
    ("vc2010_x86", "VC++ 2010 SP1 (x86)", VC2010_X86_URL),
]


def get_runtimes() -> List[Tuple[str, str, str]]:
    is64 = platform.machine().endswith("64") or sys.maxsize > 2**32
    result = []
    result.append(("vc2015_x64", "VC++ 2015-2022 (x64)", VC2015_X64_URL))
    if is64:
        result.append(("vc2015_x86", "VC++ 2015-2022 (x86)", VC2015_X86_URL))
    result.append(("vc2010_x64", "VC++ 2010 SP1 (x64)", VC2010_X64_URL))
    if is64:
        result.append(("vc2010_x86", "VC++ 2010 SP1 (x86)", VC2010_X86_URL))
    return result


def run_setup(check_only: bool = False, silent: bool = False,
              progress_cb=None, log_cb=None, target_exe: Optional[str] = None) -> Dict[str, Tuple[bool, str]]:
    results = {}
    runtimes = get_runtimes()

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    log("=" * 50)
    log("Checking VC++ Redistributables...")
    log("=" * 50)

    all_ok = True
    for name, display_name, _ in runtimes:
        installed, status = check_vc(name)
        results[name] = (installed, status)
        log(f"  {display_name}: {'[OK]' if installed else '[MISSING]'} {status}")
        if not installed:
            all_ok = False

    if check_only:
        log("\nDone. Run without --check to install missing runtimes.")
        return results

    if all_ok:
        log("\nAll runtimes installed!")
        return results

    log("\n" + "=" * 50)
    log("Installing missing runtimes...")
    log("=" * 50)

    temp = Path(tempfile.mkdtemp(prefix="runtime_setup_"))

    for name, display_name, url in runtimes:
        installed, _ = results.get(name, (False, ""))
        if installed:
            log(f"\n[SKIP] {display_name} - already installed")
            continue

        log(f"\n[INSTALL] {display_name}")
        log(f"  Downloading...")

        installer = temp / f"{name}.exe"

        def dl_progress(d, t, p):
            if progress_cb:
                progress_cb(name, p)

        if not download(url, installer, dl_progress):
            log("  [ERROR] Download failed!")
            results[name] = (False, "Download failed")
            continue

        log(f"  Downloaded: {installer.stat().st_size / 1024 / 1024:.1f} MB")
        log(f"  Installing...")

        ok, msg = install(installer, silent=silent)
        log(f"  {msg}")

        if ok:
            installed, status = check_vc(name)
            results[name] = (installed, status)
        else:
            results[name] = (False, msg)

        try:
            installer.unlink()
        except Exception:
            pass

    try:
        temp.rmdir()
    except Exception:
        pass

    log("\n" + "=" * 50)
    log("Complete!")
    log("=" * 50)

    return results


# =============================================================================
# GUI Class
# =============================================================================

class SetupGUI:
    def __init__(self, target_exe: Optional[str] = None):
        self.target_exe = target_exe
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("550x480")
        self.root.resizable(True, True)
        self._center_window()
        self._create_widgets()
        self._check_initial()

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(f, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(pady=(0, 3))
        ttk.Label(f, text="检查并安装 VC++ 运行库", font=("Segoe UI", 9)).pack(pady=(0, 10))

        # Target exe
        if self.target_exe:
            ttk.Label(f, text=f"目标程序: {self.target_exe}", font=("Segoe UI", 10)).pack(pady=(0, 10))

        # Status
        sf = ttk.LabelFrame(f, text="运行库状态", padding=10)
        sf.pack(fill=tk.X, pady=(0, 10))
        self.status_labels = {}

        for name, display_name, _ in get_runtimes():
            r = ttk.Frame(sf)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=display_name + ":", width=22, anchor=tk.W).pack(side=tk.LEFT)
            lbl = ttk.Label(r, text="检查中...", width=18, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            self.status_labels[name] = lbl

        # Progress
        pf = ttk.LabelFrame(f, text="进度", padding=10)
        pf.pack(fill=tk.X, pady=(0, 10))
        self.progress_var = tk.StringVar(value="准备就绪")
        ttk.Label(pf, textvariable=self.progress_var).pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(pf, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Log
        lf = ttk.LabelFrame(f, text="日志", padding=10)
        lf.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.log = scrolledtext.ScrolledText(lf, height=8, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

        # Buttons
        bf = ttk.Frame(f)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="检查", command=self._check).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf, text="安装缺失项", command=self._install).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf, text="打开 Microsoft 下载页", command=lambda: webbrowser.open(
            "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf, text="关闭", command=self.root.quit).pack(side=tk.RIGHT)

    def _log(self, msg: str):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _update_status(self, name: str, installed: bool, status: str):
        if name in self.status_labels:
            text = f"✓ {status}" if installed else f"✗ {status}"
            self.status_labels[name].config(text=text, foreground="green" if installed else "red")

    def _set_progress(self, pct: int):
        self.progress_bar["value"] = pct
        self.root.update_idletasks()

    def _check_initial(self):
        self._log("正在检查...")
        results = check_all()
        for name, (inst, stat) in results.items():
            self._update_status(name, inst, stat)
        self._log("检查完成")

    def _check(self):
        self._check_initial()

    def _install(self):
        for btn in self.root.winfo_children():
            if isinstance(btn, ttk.Frame):
                for child in btn.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.config(state=tk.DISABLED)

        def prog_cb(name: str, pct: int):
            self.progress_var.set(f"安装 {name}... {pct}%")
            self._set_progress(pct)

        def log_cb(msg: str):
            self._log(msg)
            self.root.update_idletasks()

        try:
            results = run_setup(progress_cb=prog_cb, log_cb=log_cb)
            for name, (inst, stat) in results.items():
                self._update_status(name, inst, stat)
            self.progress_var.set("完成!")
            self._set_progress(100)
        except Exception as e:
            self._log(f"错误: {e}")
            self.progress_var.set("失败")
        finally:
            for btn in self.root.winfo_children():
                if isinstance(btn, ttk.Frame):
                    for child in btn.winfo_children():
                        if isinstance(child, ttk.Button):
                            child.config(state=tk.NORMAL)

    def run(self):
        self.root.mainloop()


# =============================================================================
# CLI
# =============================================================================

def run_cli(check_only: bool, silent: bool, target_exe: Optional[str]):
    print(f"\n{APP_NAME} v{VERSION}")
    print("=" * 50)

    if check_only:
        print("\n[CHECK MODE]\n")
    elif silent:
        print("\n[SILENT MODE]\n")

    def prog_cb(name, pct):
        print(f"  {name}: {pct}%", end="\r")

    def log_cb(msg):
        print(msg)

    results = run_setup(check_only, silent, prog_cb, log_cb)

    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)

    ok, miss = 0, 0
    for name, (inst, _) in results.items():
        print(f"  {name}: {'INSTALLED' if inst else 'MISSING'}")
        if inst:
            ok += 1
        else:
            miss += 1

    print(f"\nTotal: {ok} installed, {miss} missing")
    return 0 if miss == 0 else 1


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Universal VC++ Runtime Installer")
    parser.add_argument("--cli", action="store_true", help="CLI mode")
    parser.add_argument("--check", action="store_true", help="Check only")
    parser.add_argument("--silent", action="store_true", help="Silent install")
    parser.add_argument("--exe", default=None, help="Target executable name")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} v{VERSION}")

    args = parser.parse_args()

    if sys.platform != "win32":
        print("Windows only.")
        return 1

    if args.cli:
        return run_cli(args.check, args.silent, args.exe)

    if not HAS_TK:
        print("tkinter not available, using CLI mode.")
        return run_cli(args.check, args.silent, args.exe)

    gui = SetupGUI(target_exe=args.exe)
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())