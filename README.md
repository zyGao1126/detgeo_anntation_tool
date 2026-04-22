# DetGeo Annotation Tool

`DetGeo Annotation Tool` 是一个面向 DetGeo / CVOGL 二次加工的桌面标注工具原型，使用 `Python + PySide6 + SQLite` 实现，目标是把 matched UAV-satellite pair 扩展为以下 query 类型：

- `Both-Exist-Single`
- `Both-Exist-Multi`
- `UAV-Only-Single`
- `UAV-Only-Multi`
- `Neither-Exist`
- `Partial-Match`

当前版本提供：

- Pair 导入与导航
- 简化后的 case-based 标注流程
- UAV 画框标注
- Satellite 手动画点成 polygon mask
- 启动时预加载 `SAM3`
- Satellite 右键切换 `SAM3` / 手动标注
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

## 安装

当前环境中未检测到 `PySide6`，因此 GUI 未在本机直接启动验证。安装依赖后即可运行：

```bash
cd /home/gaoziyang/research/CVOGL/detgeo_annotation_tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果是在 Linux 的 conda 环境中启动 GUI，建议补齐 Qt 的 `xcb` 运行时依赖：

```bash
conda activate dataAnno
conda install -c conda-forge xcb-util-cursor
```

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

```bash
python app.py gui \
  --db /home/gaoziyang/research/CVOGL/detgeo_annotation_tool/workspace/annotator.db \
  --workspace /home/gaoziyang/research/CVOGL/detgeo_annotation_tool/workspace
```

## 当前交互

- 先在右侧新建一个 `Case`
- `Category` 为下拉框，默认会按 DetGeo 原始类别自动映射
- 打开 `Annotation Mode`
- 点击 UAV 图后直接拖框
- 在 Satellite 图上右键选择 `SAM3标注` 或 `手动标注`
- `SAM3标注` 下拖框，等待 mask 预览，再按 `Ctrl+S`
- `手动标注` 下左键连续点选边界，双击闭合，或直接 `Ctrl+S`
- `Esc` 清空当前草稿
- `Show DetGeo Reference` 只用于参考，不写入当前 case
- UAV 图右键可选择 `下载图像`，默认文件名使用当前 pair 的 UAV 图名
- `Save Anno` 会把当前 case 落盘到 `saved_anno/`

## 快捷键

- `Ctrl+S`: 保存当前草稿或当前 case 元信息
- `Esc`: 清空当前 viewer 的草稿
- `Delete`: 删除当前 case

## 说明

- 当前 GUI 已简化为基于 `Case` 的独立标注，每个 case 对应同一 pair 下的一种情况，彼此互不影响。
- 相同图像对可以创建多个 case，导出时会分别复制原图与对应 mask。
- `SAM3` 在 GUI 启动阶段即进行预加载，避免第一次标注时再等待模型初始化。
- `UAV Viewer` 右键支持导出当前带框图像，导出目录会记在 `workspace/ui_state.json`。
- `SAM3` 通过 `/home/gaoziyang/research/RRSIS/sam3` 与 `/home/gaoziyang/research/RRSIS/sam3/checkpoints/sam3.pt` 调用。
