import numpy as np
import sympy as sp
from surface_integral.utils import numeric_flux_integral, parse_scalar_function

# 示例 2 的参数
u_sym, v_sym = sp.symbols('u v')
x_expr = u_sym * sp.cos(v_sym)
y_expr = u_sym**2
z_expr = u_sym * sp.sin(v_sym)

# 创建参数化函数
param_map = sp.lambdify((u_sym, v_sym), (x_expr, y_expr, z_expr), modules=['numpy'])

# 向量场
P_callable, _, _ = parse_scalar_function("x*(z**2)")
Q_callable, _, _ = parse_scalar_function("1/y")
R_callable, _, _ = parse_scalar_function("(x**2)*z")

# 范围
a, b = 1.0, float(np.sqrt(2))
c, d = 0.0, 2*np.pi

print("=" * 60)
print("测试示例 2：第二型曲面积分")
print("=" * 60)
print(f"参数化: x=u*cos(v), y=u², z=u*sin(v)")
print(f"向量场: P=x*z², Q=1/y, R=x²*z")
print(f"范围: u∈[{a:.6f}, {b:.6f}], v∈[0, 2π]")
print(f"理论解析解: π(7/6-ln2) ≈ 1.487605")
print("-" * 60)

# 测试不同网格精度
for n in [50, 100, 150, 200]:
    flux = numeric_flux_integral(P_callable, Q_callable, R_callable, ((a, b), (c, d)), param_map, nu=n, nv=n)
    error = abs(flux - 1.487605)
    print(f"网格 {n}×{n}: 数值通量 = {flux:.6f}, 误差 = {error:.6f}")

print("=" * 60)

# 手动验证法向量计算
print("\n手动验证（取u=1.2, v=π/4处的点）:")
u_test, v_test = 1.2, np.pi/4
x_test, y_test, z_test = param_map(u_test, v_test)
print(f"曲面点: ({x_test:.4f}, {y_test:.4f}, {z_test:.4f})")

# 计算偏导数（解析）
rx_u = sp.diff(x_expr, u_sym)
ry_u = sp.diff(y_expr, u_sym)
rz_u = sp.diff(z_expr, u_sym)
rx_v = sp.diff(x_expr, v_sym)
ry_v = sp.diff(y_expr, v_sym)
rz_v = sp.diff(z_expr, v_sym)

print(f"r_u = ({rx_u}, {ry_u}, {rz_u})")
print(f"r_v = ({rx_v}, {ry_v}, {rz_v})")

# 在测试点计算
ru_val = [float(rx_u.subs([(u_sym, u_test), (v_sym, v_test)])),
          float(ry_u.subs([(u_sym, u_test), (v_sym, v_test)])),
          float(rz_u.subs([(u_sym, u_test), (v_sym, v_test)]))]
rv_val = [float(rx_v.subs([(u_sym, u_test), (v_sym, v_test)])),
          float(ry_v.subs([(u_sym, u_test), (v_sym, v_test)])),
          float(rz_v.subs([(u_sym, u_test), (v_sym, v_test)]))]

print(f"r_u(测试点) = ({ru_val[0]:.4f}, {ru_val[1]:.4f}, {ru_val[2]:.4f})")
print(f"r_v(测试点) = ({rv_val[0]:.4f}, {rv_val[1]:.4f}, {rv_val[2]:.4f})")

# 叉积
n_val = [ru_val[1]*rv_val[2] - ru_val[2]*rv_val[1],
         ru_val[2]*rv_val[0] - ru_val[0]*rv_val[2],
         ru_val[0]*rv_val[1] - ru_val[1]*rv_val[0]]
print(f"n = r_u × r_v = ({n_val[0]:.4f}, {n_val[1]:.4f}, {n_val[2]:.4f})")

# 向量场值
P_val = float(P_callable(x_test, y_test, z_test))
Q_val = float(Q_callable(x_test, y_test, z_test))
R_val = float(R_callable(x_test, y_test, z_test))
print(f"F = ({P_val:.4f}, {Q_val:.4f}, {R_val:.4f})")

# 点积
dot_product = P_val * n_val[0] + Q_val * n_val[1] + R_val * n_val[2]
print(f"F · n = {dot_product:.4f}")