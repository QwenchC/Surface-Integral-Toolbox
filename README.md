
曲面积分工具箱

快速开始

1) 创建并激活 Python 虚拟环境，然后安装依赖:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) 在本机启动 Streamlit 应用（打开浏览器，默认端口 31415）:

```powershell
python -m streamlit run app.py --server.port 31415
```

主要功能

- 支持第一型曲面积分（\u2228 f dS）和第二型（通量）曲面积分。
- 优先尝试解析解（SymPy），失败时回退到数值积分（NumPy）。
- 可视化参数化曲面并可选叠加向量场（箭头为短小且浅色以减少遮挡）。

示例（可直接在应用中点击加载）

- 第一型（单位面积积分）
  - 参数化: x = u, y = v, z = u**2 + v**2
  - u range: 0,1
  - v range: 0,1
  - f(x,y,z): 1

- 第二型（通量）示例：球面上的通量
  - 参数化: x = cos(u)*sin(v), y = sin(u)*sin(v), z = cos(v)
  - u range: 0,2*pi
  - v range: 0,pi
  - P, Q, R: x, y, z

注意事项

- 范围（u/v）可输入像 `sqrt(2)`、`pi` 等 SymPy 可识别的表达式。范围用逗号分隔，例如 `0,pi/2`。
- 向量场分量请使用 Python / SymPy 风格的表达式，例如 `x*z**2` 或 `x*y`。
- 当数值结果和解析结果不一致时，可尝试提高 "网格分辨率"（应用中有网格参数）以提高数值精度。

将更改提交到 GitHub

如果你已经有一个远程仓库并且配置了 `origin`，可以使用以下命令将修改提交并推送到远程：

```powershell
git checkout -b update/readme
git add README.md
git commit -m "docs: improve README and usage instructions"
git push -u origin update/readme
```

如果没有远程仓库，请先在 GitHub 上创建一个新仓库，然后将它添加为远程：

```powershell
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin update/readme
```

我可以在本地为你执行分支创建、提交和推送（需要你的远程 URL 或已配置的 `origin` 权限）。如果你想让我现在执行，请回复并提供远程仓库 URL 或确认使用当前仓库的 `origin`。

