import streamlit as st
import sympy as sp
import numpy as np
from surface_integral.utils import parse_scalar_function, try_symbolic_surface_integral, numeric_surface_integral
import plotly.graph_objects as go


st.set_page_config(page_title="曲面积分工具箱", layout="wide")

st.title("曲面积分工具箱 - 3D")

# --- 示例数据 ---
EXAMPLES = {
    "example1": {
        "integral_type_idx": 0,
        "func": "1",
        "P": "0",
        "Q": "0",
        "R": "0",
        "xu": "cos(u)*sin(v)",
        "yu": "sin(u)*sin(v)",
        "zu": "cos(v)",
        "urange": "0,2*pi",
        "vrange": "0,pi"
    },
    "example2": {
        "integral_type_idx": 1,
        "func": "1",
        "P": "x*(z**2)",
        "Q": "1/y",
        "R": "(x**2)*z",
        "xu": "u*cos(v)",
        "yu": "u**2",
        "zu": "u*sin(v)",
        "urange": "1,sqrt(2)",
        "vrange": "0,2*pi"
    }
}

# --- 初始化持久状态 ---
if 'integral_type_idx' not in st.session_state:
    st.session_state.integral_type_idx = 0
if 'func' not in st.session_state:
    st.session_state.func = "1"
if 'P' not in st.session_state:
    st.session_state.P = "0"
if 'Q' not in st.session_state:
    st.session_state.Q = "0"
if 'R' not in st.session_state:
    st.session_state.R = "0"
if 'xu' not in st.session_state:
    st.session_state.xu = "u"
if 'yu' not in st.session_state:
    st.session_state.yu = "v"
if 'zu' not in st.session_state:
    st.session_state.zu = "u**2+v**2"
if 'urange' not in st.session_state:
    st.session_state.urange = "0,1"
if 'vrange' not in st.session_state:
    st.session_state.vrange = "0,1"

# --- 左右分栏布局：左侧 1/3 输入，右侧 2/3 显示 ---
col_input, col_display = st.columns([1, 2])

with col_input:
    st.subheader("输入区域")
    
    # 使用 session_state 持久化选择
    type_options = ["第一型：∬ f(x,y,z) dS", "第二型：∬ P dy dz + Q dz dx + R dx dy"]
    
    integral_type = st.selectbox(
        "选择积分类型", 
        type_options,
        index=st.session_state.integral_type_idx
    )
    
    # 同步 integral_type_idx（在 selectbox 值变化时）
    current_idx = type_options.index(integral_type)
    st.session_state.integral_type_idx = current_idx

    if integral_type.startswith("第一型"):
        st.markdown("**第一型曲面积分（面积分）**")
        func = st.text_input(
            "被积函数 f(x,y,z)", 
            value=st.session_state.func
        )
        # 同步到 session_state
        st.session_state.func = func
    else:
        st.markdown("**第二型曲面积分（通量）**")
        P = st.text_input("P(x,y,z)", value=st.session_state.P)
        Q = st.text_input("Q(x,y,z)", value=st.session_state.Q)
        R = st.text_input("R(x,y,z)", value=st.session_state.R)
        
        # 同步到 session_state
        st.session_state.P = P
        st.session_state.Q = Q
        st.session_state.R = R

    st.markdown("---")
    st.markdown("**积分区域（参数化）**")
    st.caption("输入参数化曲面：x(u,v), y(u,v), z(u,v) 以及 u,v 范围")
    
    xu = st.text_input("x(u,v)", value=st.session_state.xu)
    yu = st.text_input("y(u,v)", value=st.session_state.yu)
    zu = st.text_input("z(u,v)", value=st.session_state.zu)
    urange = st.text_input("u range (a,b)", value=st.session_state.urange, help="支持表达式，如: 1,sqrt(2)")
    vrange = st.text_input("v range (c,d)", value=st.session_state.vrange, help="支持表达式，如: 0,2*pi")

    # 同步到 session_state
    st.session_state.xu = xu
    st.session_state.yu = yu
    st.session_state.zu = zu
    st.session_state.urange = urange
    st.session_state.vrange = vrange

    show_vector = st.checkbox("显示向量场（第二型）", value=True)
    
    compute_btn = st.button("绘制并计算", type="primary", use_container_width=True)
    
    # --- 示例区域 ---
    st.markdown("---")
    with st.expander("📋 示例模板", expanded=False):
        st.markdown("**示例 1：球面第一型**")
        st.code("""积分类型: 第一型
f(x,y,z): 1
x(u,v): cos(u)*sin(v)
y(u,v): sin(u)*sin(v)
z(u,v): cos(v)
u range: 0, 2*pi
v range: 0, pi""", language=None)
        
        def load_example1():
            ex = EXAMPLES["example1"]
            st.session_state.integral_type_idx = ex["integral_type_idx"]
            st.session_state.func = ex["func"]
            st.session_state.P = ex["P"]
            st.session_state.Q = ex["Q"]
            st.session_state.R = ex["R"]
            st.session_state.xu = ex["xu"]
            st.session_state.yu = ex["yu"]
            st.session_state.zu = ex["zu"]
            st.session_state.urange = ex["urange"]
            st.session_state.vrange = ex["vrange"]
        
        if st.button("📥 加载示例 1", use_container_width=True, key="load_ex1"):
            load_example1()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**示例 2：第二型通量积分**")
        st.code("""积分类型: 第二型
P(x,y,z): x*(z**2)
Q(x,y,z): 1/y
R(x,y,z): (x**2)*z
x(u,v): u*cos(v)
y(u,v): u**2
z(u,v): u*sin(v)
u range: 1, sqrt(2)
v range: 0, 2*pi""", language=None)
        
        def load_example2():
            ex = EXAMPLES["example2"]
            st.session_state.integral_type_idx = ex["integral_type_idx"]
            st.session_state.func = ex["func"]
            st.session_state.P = ex["P"]
            st.session_state.Q = ex["Q"]
            st.session_state.R = ex["R"]
            st.session_state.xu = ex["xu"]
            st.session_state.yu = ex["yu"]
            st.session_state.zu = ex["zu"]
            st.session_state.urange = ex["urange"]
            st.session_state.vrange = ex["vrange"]
        
        if st.button("📥 加载示例 2", use_container_width=True, key="load_ex2"):
            load_example2()
            st.rerun()

# --- 右侧显示区域 ---
with col_display:
    st.subheader("可视化与结果")
    
    if compute_btn:
        try:
            u_sym, v_sym = sp.symbols('u v')
            x_expr = sp.sympify(xu)
            y_expr = sp.sympify(yu)
            z_expr = sp.sympify(zu)

            # 解析 u,v 范围，支持像 "1,sqrt(2)"、"0,pi" 等 SymPy 表达式
            def _parse_interval(s):
                parts = [p.strip() for p in s.split(',')]
                if len(parts) != 2:
                    raise ValueError("范围应为两个逗号分隔的值，例如: 0,1 或 1,sqrt(2)")
                vals = []
                for part in parts:
                    try:
                        symv = sp.sympify(part)
                        num = float(sp.N(symv))
                    except Exception as e:
                        raise ValueError(f"无法解析范围值 '{part}': {e}")
                    vals.append(num)
                return vals[0], vals[1]

            a, b = _parse_interval(urange)
            c, d = _parse_interval(vrange)

            # create param map
            param_map = sp.lambdify((u_sym, v_sym), (x_expr, y_expr, z_expr), modules=['numpy'])

            # sample for plotting
            us = np.linspace(a, b, 60)
            vs = np.linspace(c, d, 60)
            U, V = np.meshgrid(us, vs, indexing='ij')
            Xn, Yn, Zn = param_map(U, V)

            fig = go.Figure(data=[go.Surface(x=Xn, y=Yn, z=Zn, opacity=0.9, name="曲面")])

            if integral_type.startswith("第一型"):
                f_callable, f_sym, _ = parse_scalar_function(func)

                # try symbolic
                ru = sp.Matrix([sp.diff(x_expr, u_sym), sp.diff(y_expr, u_sym), sp.diff(z_expr, u_sym)])
                rv = sp.Matrix([sp.diff(x_expr, v_sym), sp.diff(y_expr, v_sym), sp.diff(z_expr, v_sym)])
                dS = sp.simplify(sp.sqrt((ru.cross(rv)).dot(ru.cross(rv))))

                integrand = sp.simplify(f_sym * dS)
                sym_res = try_symbolic_surface_integral(None, integrand, (u_sym, a, b, v_sym, c, d))
                if sym_res is not None:
                    st.success(f"✓ 解析解: {sym_res}")
                else:
                    # numeric
                    def param_map_np(u, v):
                        vals = param_map(u, v)
                        return np.array(vals)

                    res = numeric_surface_integral(f_callable, ((a, b), (c, d)), param_map_np)
                    st.info(f"≈ 数值解(近似): {res:.6f}")

                fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, key="plot_type1")

            else:
                # 第二型：计算通量 + 可选向量场
                try:
                    P_callable, P_sym, _ = parse_scalar_function(P)
                    Q_callable, Q_sym, _ = parse_scalar_function(Q)
                    R_callable, R_sym, _ = parse_scalar_function(R)

                    def param_map_np(u, v):
                        vals = param_map(u, v)
                        return np.array(vals)

                    from surface_integral.utils import numeric_flux_integral

                    flux = numeric_flux_integral(P_callable, Q_callable, R_callable, ((a, b), (c, d)), param_map_np)
                    st.success(f"✓ 数值通量(近似): {flux:.6f}")

                    if show_vector:
                        # --- 向量场：在整个显示区域按 0.25 步长 3D 格点采样 ---
                        all_x = np.asarray(Xn, dtype=float).ravel()
                        all_y = np.asarray(Yn, dtype=float).ravel()
                        all_z = np.asarray(Zn, dtype=float).ravel()

                        xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
                        ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
                        zmin, zmax = float(np.nanmin(all_z)), float(np.nanmax(all_z))

                        # 扩展包围盒以覆盖显示区域
                        xrange = max(1e-6, xmax - xmin)
                        yrange = max(1e-6, ymax - ymin)
                        zrange = max(1e-6, zmax - zmin)
                        margin = 0.25
                        xmin -= xrange * margin; xmax += xrange * margin
                        ymin -= yrange * margin; ymax += yrange * margin
                        zmin -= zrange * margin; zmax += zrange * margin

                        step = 0.25

                        # 确保每个维度至少两个采样点
                        def _grid(lo, hi, step):
                            g = np.arange(lo, hi + 1e-9, step, dtype=float)
                            if g.size < 2:
                                g = np.array([lo - step, hi + step], dtype=float)
                            return g

                        gx = _grid(xmin, xmax, step)
                        gy = _grid(ymin, ymax, step)
                        gz = _grid(zmin, zmax, step)

                        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
                        sxp = GX.ravel()
                        syp = GY.ravel()
                        szp = GZ.ravel()

                        # 点太多会卡：做一个上限裁剪（保持网格均匀抽样）
                        max_points = 5000
                        npts = sxp.size
                        if npts > max_points:
                            idx = np.linspace(0, npts - 1, max_points).astype(int)
                            sxp, syp, szp = sxp[idx], syp[idx], szp[idx]

                        def _eval_field(callable_f, x, y, z):
                            x = np.asarray(x, dtype=float)
                            y = np.asarray(y, dtype=float)
                            z = np.asarray(z, dtype=float)
                            try:
                                out = callable_f(x, y, z)
                                out = np.asarray(out, dtype=float)
                                if out.shape != x.shape:
                                    out = np.vectorize(lambda xi, yi, zi: float(callable_f(xi, yi, zi)))(x, y, z)
                                return out
                            except Exception:
                                return np.asarray([float(callable_f(xi, yi, zi)) for xi, yi, zi in zip(x.ravel(), y.ravel(), z.ravel())], dtype=float).reshape(x.shape)

                        Vx = _eval_field(P_callable, sxp, syp, szp)
                        Vy = _eval_field(Q_callable, sxp, syp, szp)
                        Vz = _eval_field(R_callable, sxp, syp, szp)

                        # 把 NaN -> 0；把 inf 替换为与已有有限值相近的最大有限值
                        def _fix_inf_nan(arr):
                            arr = np.asarray(arr, dtype=float)
                            arr = np.nan_to_num(arr, nan=0.0)
                            inf_mask = np.isinf(arr)
                            if inf_mask.any():
                                finite = arr[np.isfinite(arr)]
                                if finite.size > 0:
                                    repl = np.max(np.abs(finite))
                                    if repl < 1e-6:
                                        repl = 1.0
                                else:
                                    repl = 1.0
                                arr[inf_mask] = np.sign(arr[inf_mask]) * repl
                            return arr

                        Vx = _fix_inf_nan(Vx)
                        Vy = _fix_inf_nan(Vy)
                        Vz = _fix_inf_nan(Vz)

                        # 向量缩短并减小箭头尺寸
                        mag = np.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
                        max_mag = float(np.max(mag)) if mag.size else 0.0
                        target = step * 0.15
                        if max_mag > 1e-12:
                            s = target / max_mag
                            Vx, Vy, Vz = Vx * s, Vy * s, Vz * s

                        cone = go.Cone(
                            x=sxp, y=syp, z=szp,
                            u=Vx, v=Vy, w=Vz,
                            anchor="tail",
                            sizemode="absolute",
                            sizeref=step * 0.25,
                            colorscale=[
                                [0.0, "rgba(200,230,255,0.5)"],
                                [1.0, "rgba(80,160,240,0.5)"],
                            ],
                            showscale=False,
                            opacity=0.85,
                            name="向量场"
                        )
                        fig.add_trace(cone)

                    fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True, key="plot_type2")

                except Exception as e:
                    st.error(f"第二型求解出错: {e}")

        except Exception as e:
            st.error(f"出错: {e}")
    else:
        st.info("👈 请在左侧输入参数并点击「绘制并计算」")
