"""数学工具库 (Math Utilities)

ODE求解器、Hill方程、Logistic/Gompertz解析解等。
"""
import math
from typing import Callable, List, Tuple, Dict, Any


def euler_step(f: Callable, y: float, t: float, dt: float, **kwargs) -> float:
    """Euler法单步ODE求解

    dy/dt = f(y, t, **kwargs)
    """
    return y + f(y, t, **kwargs) * dt


def rk4_step(f: Callable, y: float, t: float, dt: float, **kwargs) -> float:
    """四阶Runge-Kutta单步ODE求解"""
    k1 = f(y, t, **kwargs)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = f(y + dt * k3, t + dt, **kwargs)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def solve_ode(
    f: Callable,
    y0: float,
    t_span: Tuple[float, float],
    dt: float = 0.1,
    method: str = "rk4",
) -> Tuple[List[float], List[float]]:
    """求解ODE

    Args:
        f: dy/dt = f(y, t)
        y0: 初始值
        t_span: (t_start, t_end)
        dt: 步长
        method: "euler" | "rk4"

    Returns:
        (times, values)
    """
    step_fn = rk4_step if method == "rk4" else euler_step
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    times = [t_start]
    values = [y0]
    y = y0
    t = t_start

    for _ in range(n_steps):
        y = step_fn(f, y, t, dt)
        t += dt
        times.append(t)
        values.append(y)

    return times, values


def logistic_growth(V: float, t: float, r: float = 0.027, K: float = 1000.0) -> float:
    """Logistic增长 dV/dt = r*V*(1 - V/K)"""
    return r * V * (1 - V / K)


def gompertz_growth(V: float, t: float, r: float = 0.027, K: float = 1000.0) -> float:
    """Gompertz增长 dV/dt = r*V*ln(K/V)"""
    if V <= 0 or K <= 0:
        return 0.0
    return r * V * math.log(K / V)


def logistic_analytical(t: float, r: float, K: float, V0: float) -> float:
    """Logistic解析解 V(t) = K / (1 + (K/V0 - 1) * exp(-r*t))"""
    return K / (1 + (K / V0 - 1) * math.exp(-r * t))


def gompertz_analytical(t: float, r: float, K: float, V0: float) -> float:
    """Gompertz解析解 V(t) = K * exp(ln(V0/K) * exp(-r*t))"""
    return K * math.exp(math.log(V0 / K) * math.exp(-r * t))


def hill_equation(C: float, EC50: float, Emax: float = 1.0, n: float = 1.0) -> float:
    """Hill方程 E = Emax * C^n / (EC50^n + C^n)"""
    if C <= 0:
        return 0.0
    Cn = C ** n
    EC50n = EC50 ** n
    return Emax * Cn / (EC50n + Cn)


def shannon_diversity(fractions: List[float]) -> float:
    """Shannon多样性指数 H = -sum(p * ln(p))"""
    H = 0.0
    for p in fractions:
        if p > 0:
            H -= p * math.log(p)
    return H


def clamp(value: float, lo: float, hi: float) -> float:
    """范围钳制"""
    return max(lo, min(hi, value))


def linear_interpolation(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """线性插值"""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def doubling_time(growth_rate: float) -> float:
    """倍增时间 Td = ln(2) / r"""
    if growth_rate <= 0:
        return float('inf')
    return math.log(2) / growth_rate