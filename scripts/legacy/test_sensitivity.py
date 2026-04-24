#!/usr/bin/env python3
import os
import sys
import json

import numpy as np

# Ensure project src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.epitope.sensitivity import SensitivityResult, sensitivity_report, format_sensitivity_report


def main():
    feature_names = [
        "aac_A",
        "aac_C",
        "aac_D",
        "hydropathy_mean_mid",
        "hydropathy_mean_n",
        "hydropathy_mean_c",
        "frac_nonpolar_mid",
        "frac_nonpolar_n",
        "env_pH",
        "env_temp",
        "other_1",
        "other_2",
    ]

    grad = np.array([0.2, -0.1, 0.0, 0.5, 0.05, -0.02, 0.3, -0.05, 0.1, -0.08, 0.0, 0.01], dtype=np.float32)
    imp = np.abs(grad)

    res = SensitivityResult(pred=0.73, grad=grad, importance=imp, feature_names=feature_names)

    report = sensitivity_report(res, top_k=6)

    print("=== Formatted Summary ===")
    print(format_sensitivity_report(report, top_k=6))

    print("\n=== JSON Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
