from typing import Tuple, Optional, Callable
import sympy as sp
import numpy as np


def parse_scalar_function(expr_str):
    """解析标量函数表达式"""
    x, y, z = sp.symbols('x y z')
    expr = sp.sympify(expr_str)
    f_sym = expr
    f_callable = sp.lambdify((x, y, z), expr, modules=['numpy'])
    return f_callable, f_sym, (x, y, z)


def try_symbolic_surface_integral(f_sym, integrand, bounds):
    """尝试符号积分"""
    try:
        u_sym, a, b, v_sym, c, d = bounds
        result = sp.integrate(integrand, (u_sym, a, b), (v_sym, c, d))
        if result.has(sp.Integral):
            return None
        return result
    except Exception:
        return None


def _trapz(y, x=None, axis=-1):
    """梯形法则积分"""
    y = np.asanyarray(y)
    if x is None:
        if y.size == 0:
            return 0.0
        slicer1 = [slice(None)] * y.ndim
        slicer2 = [slice(None)] * y.ndim
        slicer1[axis] = slice(1, None)
        slicer2[axis] = slice(0, -1)
        y1 = y[tuple(slicer1)]
        y0 = y[tuple(slicer2)]
        return np.sum((y0 + y1) / 2.0, axis=axis)
    else:
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError("x must be one-dimensional")
        if x.size < 2:
            return 0.0
        dx = x[1:] - x[:-1]
        slicer1 = [slice(None)] * y.ndim
        slicer2 = [slice(None)] * y.ndim
        slicer1[axis] = slice(1, None)
        slicer2[axis] = slice(0, -1)
        y1 = y[tuple(slicer1)]
        y0 = y[tuple(slicer2)]
        shape = [1] * y.ndim
        shape[axis] = dx.size
        dx_shape = dx.reshape(shape)
        return np.sum((y0 + y1) / 2.0 * dx_shape, axis=axis)


def numeric_surface_integral(f_callable, bounds, param_map, nu=100, nv=100):
    """第一型曲面积分的数值计算"""
    (a, b), (c, d) = bounds
    us = np.linspace(a, b, nu)
    vs = np.linspace(c, d, nv)
    U, V = np.meshgrid(us, vs, indexing='ij')
    
    # 计算曲面点
    X, Y, Z = param_map(U, V)
    
    # 计算偏导数（用差分近似）
    du = (b - a) / (nu - 1) if nu > 1 else 1.0
    dv = (d - c) / (nv - 1) if nv > 1 else 1.0
    
    # r_u: ∂r/∂u
    rx_u = np.gradient(X, du, axis=0)
    ry_u = np.gradient(Y, du, axis=0)
    rz_u = np.gradient(Z, du, axis=0)
    
    # r_v: ∂r/∂v
    rx_v = np.gradient(X, dv, axis=1)
    ry_v = np.gradient(Y, dv, axis=1)
    rz_v = np.gradient(Z, dv, axis=1)
    
    # 计算叉积 r_u × r_v
    nx = ry_u * rz_v - rz_u * ry_v
    ny = rz_u * rx_v - rx_u * rz_v
    nz = rx_u * ry_v - ry_u * rx_v
    
    # dS = |r_u × r_v|
    dS = np.sqrt(nx**2 + ny**2 + nz**2)
    
    # 计算被积函数值
    f_vals = f_callable(X, Y, Z)
    
    # 数值积分
    integrand = f_vals * dS
    result = _trapz(_trapz(integrand, vs, axis=1), us, axis=0)
    
    return float(result)


def numeric_flux_integral(P_callable, Q_callable, R_callable, bounds, param_map, nu=100, nv=100):
    """
    第二型曲面积分（通量积分）的数值计算
    ∬ P dy∧dz + Q dz∧dx + R dx∧dy
    = ∬ (P, Q, R) · (∂r/∂u × ∂r/∂v) du dv
    """
    (a, b), (c, d) = bounds
    us = np.linspace(a, b, nu)
    vs = np.linspace(c, d, nv)
    U, V = np.meshgrid(us, vs, indexing='ij')
    
    # 计算曲面点
    X, Y, Z = param_map(U, V)
    
    # 计算偏导数
    du = (b - a) / (nu - 1) if nu > 1 else 1.0
    dv = (d - c) / (nv - 1) if nv > 1 else 1.0
    
    # r_u: ∂r/∂u
    rx_u = np.gradient(X, du, axis=0)
    ry_u = np.gradient(Y, du, axis=0)
    rz_u = np.gradient(Z, du, axis=0)
    
    # r_v: ∂r/∂v
    rx_v = np.gradient(X, dv, axis=1)
    ry_v = np.gradient(Y, dv, axis=1)
    rz_v = np.gradient(Z, dv, axis=1)
    
    # 法向量 n = r_u × r_v (注意：这里的顺序决定了法向量方向)
    nx = ry_u * rz_v - rz_u * ry_v
    ny = rz_u * rx_v - rx_u * rz_v
    nz = rx_u * ry_v - ry_u * rx_v
    
    # 计算向量场在曲面上的值
    P_vals = P_callable(X, Y, Z)
    Q_vals = Q_callable(X, Y, Z)
    R_vals = R_callable(X, Y, Z)
    
    # 处理 inf/NaN
    P_vals = np.nan_to_num(P_vals, nan=0.0, posinf=1e10, neginf=-1e10)
    Q_vals = np.nan_to_num(Q_vals, nan=0.0, posinf=1e10, neginf=-1e10)
    R_vals = np.nan_to_num(R_vals, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # 点积：F · n = P*nx + Q*ny + R*nz
    integrand = P_vals * nx + Q_vals * ny + R_vals * nz
    
    # 数值积分
    result = _trapz(_trapz(integrand, vs, axis=1), us, axis=0)
    
    return float(result)

