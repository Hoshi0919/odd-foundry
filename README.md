# Odd Foundry (奇弹工场)

> **Active Odd Elasticity, Non-Reciprocal Metamaterials & Anti-Jamming Logistics Simulator**  
> 基于奇弹性（Odd Elasticity）与非互易力学超构材料的自运转物理工业模拟器。

[![CI Tests](https://img.shields.io/badge/tests-23%2F23%20passed-10b981.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-0284c7.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-00f2fe.svg)](https://hoshi0919.github.io/odd-foundry/)

---

## 🌟 核心理念与物理前沿

传统工厂与建造类模拟（如 *Factorio*、*Minecraft Create*）均建立在经典牛顿力学与格林超弹性（Green Elasticity）的互易性基础之上：
- 任何做功循环必须依赖外源电机旋转驱动；
- 弹簧受力遵守胡克定律与对称性：$F = -kx$；
- 狭窄漏斗咽喉因库仑摩擦与法向挤压必然形成静态卡料拱（Janssen 效应）；
- 冲击波在机械管道中双向传播，下游碰撞会震损上游精密装配线。

**Odd Foundry（奇弹工场）** 彻底推翻了这一传统经典机械范式，将近五年来凝聚态物理与活性物质领域的革命性发现——**奇弹性（Odd Elasticity, Scheibner et al., Nature Physics 2020）** 与 **非互易力学超构材料（Non-Reciprocal Metamaterials）** 引入宏观自动化工业建造网络：

$$\begin{pmatrix} \sigma_v \\ \sigma_s \\ \sigma_a \end{pmatrix} = \begin{pmatrix} K & 0 & 0 \\ 0 & \mu & A_o \\ 0 & -A_o & \mu \end{pmatrix} \begin{pmatrix} \epsilon_v \\ \epsilon_s \\ \epsilon_a \end{pmatrix}$$

1. **破缺麦克斯韦-贝蒂互易性**：弹性张量反对称模量 $A_o \neq 0$；
2. **应变循环做正功**：闭合应变轨迹做功 $\oint \boldsymbol{\sigma} : d\boldsymbol{\epsilon} = 2\pi A_o \epsilon_0^2 > 0$；超构材料本身即是持续供能的发电机；
3. **自发手性行波单向巡航**：表层质点逆行椭圆共振，物料无马达、无轮轨自发沿波峰向前流动；
4. **自愈反卡料漏斗**：法向挤压诱发正交奇剪切力，瞬时瓦解静态摩擦力拱，实现无堵塞流体化下料；
5. **非互易声学二极管**：空间宇称与时间反演联合破缺，正向波无损穿透，反向冲击波指数淬灭（隔离度 $> 20\text{ dB}$）。

---

## 🔬 四大动力学机制

### 1. 自运转非互易行波传输带 (`WaveConveyor`)
表层节点在奇弹性作用下自发维持手性逆行椭圆振颤：
$$u_x(x, t) = A_x \sin(\omega t - k x), \quad u_y(x, t) = A_y \cos(\omega t - k x)$$
在波峰（$u_y > 0$）处，节点法向推力达到峰值且横向速度 $v_x > 0$，强力推动物料向前；在波谷处脱离接触。无需机械传送带或旋转轴即可恒速巡航。

### 2. 奇应力防卡拱漏斗 (`OddHopper`)
颗粒挤压在咽喉处形成接触力链时，受法向挤压力 $F_n$ 约束：
$$\mathbf{F}_{\text{transverse}} = \eta_{\text{odd}} F_n \hat{\mathbf{t}}$$
奇剪切力打破共线力平衡，赋予桥拱主动自转力矩，在 $< 0.1\text{ s}$ 内击碎卡死结构，保持流体般平滑出料。

### 3. 循环应变动力提取仓 (`StrainEngine`)
外部施加微幅交变形变 $(\epsilon_s(t) = \epsilon_0 \cos \Omega t, \epsilon_a(t) = \epsilon_0 \sin \Omega t)$，奇弹性超构材料直接将微观非平衡能源转化为宏观机械推力，货物穿过提取仓获得确定性动能跃升：
$$v_{\text{exit}} = \sqrt{v_{\text{entry}}^2 + \frac{2 W_{\text{extracted}}}{m}}$$

### 4. 非互易机械二极管与减震波导 (`MechanicalDiode`)
一维振子链引入非对称手性耦合 $-\alpha_{\text{odd}} (u_{j+1} - u_{j-1})$，左邻耦合权重 $(k + \alpha)$，右邻耦合权重 $(k - \alpha)$：
- 正向声子透射率 $T_{\to} > 0.85$；
- 反向冲击波透射率 $T_{\leftarrow} < 0.005$；
- 隔离对比度 $> 22\text{ dB}$，下游货物急停冲击波永不回溯。

---

## 🚀 快速上手与验证

### 1. 运行测试套件 (23 Tests 100% Green)
```bash
uv run --with pytest --with numpy pytest tests/
```

### 2. 启动前端可视化原型
打开浏览器直接访问 `index.html`，或本地预览：
```bash
python3 -m http.server 8080
```
或直接访问部署好的公网 GitHub Pages：  
👉 **[https://hoshi0919.github.io/odd-foundry/](https://hoshi0919.github.io/odd-foundry/)**

---

## 📂 工程架构

```
odd-foundry/
├── odd/
│   ├── __init__.py         # 包导出与元数据
│   ├── constitutive.py     # 2D 各向同性奇弹性本构关系与应变空间热力学
│   ├── lattice.py          # 离散非互易奇弹性弹簧网络动力学
│   ├── wave_conveyor.py    # 自运转手性行波单向传输机
│   ├── anti_jamming.py     # 奇应力防卡拱漏斗颗粒流求解器
│   ├── strain_engine.py    # 循环应变动能提取与推进加速器
│   ├── diode.py            # 非厄米声学二极管与单向减震波导
│   └── engine.py           # 完整工业自运转全流程调度引擎
├── tests/
│   ├── __init__.py
│   └── test_odd_foundry.py # 23 项数学与物理严密回归测试
├── index.html              # HTML5 Canvas 交互式仿真单文件原型 (无外部依赖)
├── pyproject.toml          # PEP 621 构建配置
├── pytest.ini              # Pytest 配置
└── README.md
```

---

## 📄 许可证

MIT License © 2026 Hoshi (<hoshi0919@outlook.com>)
