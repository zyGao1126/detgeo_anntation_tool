# DetGeo Annotation Tool

`DetGeo Annotation Tool` 是一个面向 DetGeo / CVOGL 二次加工的桌面标注工具原型，使用 `Python + PySide6 + SQLite` 实现，当前围绕每个 matched UAV-satellite pair 组织 `single_target` / `multiple_target` 两类 case，并支持构造 case 级别的难负样本。

当前版本提供：

- Pair 导入与导航
- 简化后的 case-based 标注流程
- `New Case` 仅提供 `single_target` / `multiple_target`
- UAV 画框标注
- Satellite 手动画点成 polygon mask
- 启动时预加载 `SAM3`
- Satellite 右键切换 `SAM3` / 手动标注
- Satellite Viewer 支持裁剪生成 `hard negative`，并切换查看原图 / 难负样本
- DetGeo 原始参考标注显示开关
- Category 下拉框与 DetGeo 原始类别映射
- UAV 图像导出到本地目录，并记住上次导出路径
- 按 case 独立保存与导出

## 目录结构

```text
detgeo_annotation_tool/
├── .gitignore
├── app.py
├── requirements.txt
├── detgeo_annotation_tool/
│   ├── models.py
│   ├── storage.py
│   ├── repository.py
│   ├── services/
│   │   ├── exporter.py
│   │   ├── importer.py
│   │   ├── qa.py
│   │   └── segmentation.py
│   └── ui/
│       ├── main_window.py
│       └── viewer.py
└── workspace/
```

## SAM3 加载机制

当前工具里的 `SAM3` 不是通过切换到另一个独立虚拟环境来调用的，而是直接复用启动 GUI 的那个 Python 解释器。

具体流程是：

1. GUI 启动后，`MainWindow` 创建 `SAM3ProcessClient`
2. `SAM3ProcessClient` 用当前解释器 `sys.executable` 启动子进程：

```bash
python -u -m detgeo_annotation_tool.services.sam3_worker \
  --repo-root <SAM3仓库路径> \
  --checkpoint <SAM3权重路径>
```

3. `sam3_worker` 调用 `SAM3Backend.preload()`
4. `SAM3Backend` 会把 `repo_root` 加到 `sys.path`
5. 然后在当前环境中直接导入：
   - `torch`
   - `sam3.model.sam3_image_processor`
   - `sam3.model_builder`
6. 再根据本地 checkpoint 文件加载模型

因此要让 `SAM3` 正常工作，必须满足：

- 运行 GUI 的 Python 环境中，同时安装了本工具依赖和 `SAM3` 依赖
- 本地存在 `SAM3` 仓库源码
- 本地存在 `SAM3` checkpoint 文件

当前版本支持通过环境变量配置 `SAM3` 路径：

- `DETGEO_SAM3_REPO`: `SAM3` 仓库根目录
- `DETGEO_SAM3_CHECKPOINT`: `SAM3` checkpoint 文件路径
- `DETGEO_SAM3_DEVICE`: `SAM3` 推理设备，支持 `auto`、`cpu`、`cuda:N`

如果不设置 `DETGEO_SAM3_DEVICE`，GUI 启动后不会自动加载 `SAM3`，需要先在顶部 `SAM3 Device` 下拉框里手动选择设备，再启动模型。这样可以避免默认占用 `cuda:0`。

如果不设置环境变量，代码会回退到作者本机默认路径：

- `/home/gaoziyang/research/RRSIS/sam3`
- `/home/gaoziyang/research/RRSIS/sam3/checkpoints/sam3.pt`

如果你的目录结构是：

```text
~/projects/
├── detgeo_annotation_tool/
└── sam3/
```

当前版本也会优先自动发现同级的 `../sam3`，不必强制依赖环境变量。

## 安装

推荐使用同一个 `conda` 环境同时安装本工具和 `SAM3`。下面是一套别人 clone 后可以直接照着执行的流程。

### 1. 克隆仓库

```bash
mkdir -p ~/projects
cd ~/projects

git clone <your_detgeo_annotation_tool_repo_url> detgeo_annotation_tool
git clone https://github.com/facebookresearch/sam3.git
```

推荐目录结构：

```text
~/projects/
├── detgeo_annotation_tool/
└── sam3/
```

### 2. 创建 Python 环境

`SAM3` 官方 README 推荐：

- Python `3.12`
- PyTorch `2.7+`
- CUDA `12.6+`

推荐命令：

```bash
conda create -n detgeo_sam3 python=3.11 -y
conda activate detgeo_sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果你的机器 CUDA 版本不同，可以改用适配你机器的 PyTorch 版本。代码会优先用 `cuda`，否则退回 `cpu`，但 `cpu` 推理会明显更慢。

### 3. 安装本工具依赖

```bash
cd ~/projects/detgeo_annotation_tool
pip install -r requirements.txt
```

如果是在 Linux 图形界面环境启动 GUI，建议补齐 Qt 的 `xcb` 运行时依赖：

```bash
conda activate detgeo_sam3
conda install -c conda-forge xcb-util-cursor
```

### 4. 安装 SAM3 依赖

在本机这份 `SAM3` 仓库里，依赖来自它的 `pyproject.toml`。最直接的安装方式是：

```bash
cd ~/projects/sam3
pip install -e .
```

但截至目前，`SAM3` 这份仓库在实际推理时还额外依赖几个没有被完整声明的包，而且新版 `setuptools` 已移除了 `pkg_resources`，会导致加载时报：

```text
No module named 'pkg_resources'
```

因此建议紧接着补这一行：

```bash
pip install "setuptools<81" einops pycocotools psutil
```

### 5. 下载 SAM3 权重

当前工具需要一个本地 checkpoint 文件路径，例如：

```text
~/projects/sam3/checkpoints/sam3.pt
```


然后将所需的 checkpoint 下载到本地，放到例如：

```bash
mkdir -p ~/projects/sam3/checkpoints
# 你的 sam3.pt 放到这里
~/projects/sam3/checkpoints/sam3.pt
```

说明：

- 当前工具按“本地单个 checkpoint 文件”来加载模型
- 文件名不强制必须叫 `sam3.pt`
- 如果你使用别的文件名，记得通过环境变量显式指定

### 6. 配置本工具读取 SAM3 路径

在启动 GUI 之前设置：

```bash
export DETGEO_SAM3_REPO=~/projects/sam3
export DETGEO_SAM3_CHECKPOINT=~/projects/sam3/checkpoints/sam3.pt
export DETGEO_SAM3_DEVICE=cuda:1
```

建议把这几行写进 `~/.bashrc`、`~/.zshrc` 或项目启动脚本里。GUI 内也可以直接通过顶部的 `SAM3 Device` 下拉框切换设备，切换后会自动重启 `SAM3 worker`。

### 7. 验证环境

在真正启动 GUI 之前，可以先检查当前环境是否能正常导入 `SAM3`：

```bash
conda activate detgeo_sam3
python - <<'PY'
import torch
import sam3
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("sam3 import ok")
PY
```

如果这几个 import 能通过，通常说明本工具的 `SAM3 worker` 也能正常启动。

## 初始化数据库

从 CVOGL/DetGeo 的 `.pth` split 文件导入 pairs：

```bash
python app.py init-db \
  --db /home/gaoziyang/research/CVOGL/detgeo_annotation_tool/workspace/annotator.db \
  --data-root /home/gaoziyang/research/CVOGL/Datasets/CVOGL \
  --data-name CVOGL_DroneAerial \
  --splits train val test
```

也支持从 `visualize_cvogl_pairs.py` 生成的 `manifest.json` 导入：

```bash
python app.py import-manifest \
  --db /home/gaoziyang/research/CVOGL/detgeo_annotation_tool/workspace/annotator.db \
  --manifest /path/to/manifest.json
```

## 启动 GUI

先确保已经激活同一个环境，并配置好 `SAM3` 路径：

```bash
conda activate detgeo_sam3
export DETGEO_SAM3_REPO=~/projects/sam3
export DETGEO_SAM3_CHECKPOINT=~/projects/sam3/checkpoints/sam3.pt
```

然后运行：

```bash
python app.py gui \
  --db ~/projects/detgeo_annotation_tool/workspace/annotator.db \
  --workspace ~/projects/detgeo_annotation_tool/workspace
```

## 当前交互

- 先在右侧新建一个 `Case`
- `Case Type` 仅保留 `single_target` / `multiple_target`
- `Category` 为下拉框，默认会按 DetGeo 原始类别自动映射
- 打开 `Annotation Mode`
- 点击 UAV 图后直接拖框
- 在 Satellite 图上右键选择 `SAM3标注` 或 `手动标注`
- `SAM3标注` 下拖框，等待 mask 预览，再按 `Ctrl+S`
- `手动标注` 下左键连续点选边界，双击闭合，或直接 `Ctrl+S`
- 点击 `Generate Hard Negative` 后，在原始卫星图上拖框裁剪，系统会自动 resize 回原图尺寸并切到预览图
- 进入 `Hard Negative` 截图模式后，会自动缩放到整图，并用黄色框提示目标区域；拖框时裁剪区域外侧会变暗，方便判断截取范围
- `Satellite Viewer` 右上角 `<->` 可在原始卫星图和当前 case 的难负样本图之间切换
- `Satellite Viewer` 右键支持删除当前 case 的难负样本
- `Esc` 清空当前草稿
- `Show DetGeo Reference` 只用于参考，不写入当前 case
- UAV 图右键可选择 `下载图像`，默认文件名使用当前 pair 的 UAV 图名
- `Save Case Metadata` 会把当前 case 的难负样本图路径与裁剪框一起写入数据库；图像文件本体保存在 `workspace/cases/<case_id>/hard_negative.png`
- `Save Anno` 会把当前 case 的原图、mask 和难负样本图一起落盘到 `saved_anno/`

## 快捷键

- `Ctrl+S`: 保存当前草稿或当前 case 元信息
- `Esc`: 清空当前 viewer 的草稿
- `Delete`: 删除当前 case

## 说明

- 当前 GUI 已简化为基于 `Case` 的独立标注，每个 case 对应同一 pair 下的一种情况，彼此互不影响。
- 相同图像对可以创建多个 case，导出时会分别复制原图与对应 mask。
- 难负样本图像按 case 独立保存，默认文件名为 `hard_negative.png`，再次生成会覆盖该 case 的旧图。
- `SAM3` 在 GUI 启动阶段即进行预加载，避免第一次标注时再等待模型初始化。
- `UAV Viewer` 右键支持导出当前带框图像，导出目录会记在 `workspace/ui_state.json`。
- `SAM3` 更推荐通过 `DETGEO_SAM3_REPO` 与 `DETGEO_SAM3_CHECKPOINT` 配置；未设置时才回退到作者本机默认路径。
