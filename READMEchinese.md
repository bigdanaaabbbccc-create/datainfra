# Omni3D & Cube R-CNN

[![支持乌克兰](https://img.shields.io/badge/支持-乌克兰-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

**Omni3D：面向真实场景三维目标检测的大规模基准与模型**

[Garrick Brazil][gb], [Abhinav Kumar][ak], [Julian Straub][js], [Nikhila Ravi][nr], [Justin Johnson][jj], [Georgia Gkioxari][gg]

[[`项目主页`](https://garrickbrazil.com/omni3d)] [[`论文（arXiv）`](https://arxiv.org/abs/2207.10660)] [[`引用（BibTeX）`](#citing)]


<table style="border-collapse: collapse; border: none;">
<tr>
	<td width="60%">
		<p align="center">
			在 <a href="https://about.facebook.com/realitylabs/projectaria">Project Aria</a> 数据上进行零样本检测与跟踪
			<img src=".github/generalization_demo.gif" alt="Aria 演示视频"/ height="300">
		</p>
	</td>
	<td width="40%">
		<p align="center">
			在 COCO 数据上的预测结果
			<img src=".github/generalization_coco.png" alt="COCO 演示"/ height="300">
		</p>
	</td>
</tr>
</table>

<!--
## Cube R-CNN 概览
<p align="center">
<img src=".github/cubercnn_overview.jpg" alt="Cube R-CNN 概览" height="300" />
</p>
-->

## 目录

1. [依赖安装（阿里云 PPU）](#installation)
2. [演示](#demo)
3. [Omni3D 数据](#data)
4. [Cube R-CNN 训练](#training)
5. [Cube R-CNN 推理](#inference)
6. [实验结果](#results)
7. [许可证](#license)
8. [引用](#citing)


## 依赖安装（阿里云 PPU） <a name="installation"></a>

本节面向灵骏 PAI DSW 的镜像
`ppu-training:2.1.0-pytorch2.0-ppu-py38-cu121-ubuntu20.04`。
项目依赖 [Detectron2][d2]、[PyTorch][pyt]、[PyTorch3D][py3d] 和 COCO API。
原项目的 PyTorch 1.8 / torchvision 0.9.1 / CUDA 10.1 安装命令不适用于此镜像。

当前实例实测环境：

| 项目 | 检测结果 |
| --- | --- |
| 系统 / Python | Ubuntu 20.04 / Python 3.8.13 |
| 加速卡 | 2 张 PPU-ZW810E |
| PPU SDK | 2.1.0-a5f865 |
| PyTorch / torchvision | `2.0.0a0+nv2303` / `0.15.1a0`（镜像预装） |
| CUDA 兼容版本 / SDK 路径 | 12.1 / `/usr/local/PPU_SDK/CUDA_SDK` |

**验证范围：** 两张卡的矩阵计算、反向传播、torchvision NMS 和 ROIAlign 前反向已通过。
尚未安装或验证 Detectron2、PyTorch3D，也未完成 Cube R-CNN 推理或训练；以下是待实际执行验证的安装流程。

### 1. 保留镜像的 PPU 软件栈，检查设备

此镜像通过 `torch.cuda` 和 `.cuda()` 接口访问 PPU，代码中的 CUDA 接口不代表必须使用 NVIDIA 卡。
保留预装的 `torch`、`torchvision`、驱动和 SDK，不要用普通 PyPI/Conda 包覆盖它们，也不要另装 NVIDIA CUDA Toolkit。
根据[阿里云 PPU 官方说明](https://help.aliyun.com/zh/pai/use-cases/pai-pg1-getting-started-best-practices)，
含加速算子的依赖优先使用与镜像匹配的 PPU 包，缺少对应包时再尝试源码编译。

在能访问加速卡的 DSW 终端中执行：

```bash
cd /mnt/workspace/omni3d
python - <<'PY'
import sys
import torch
import torchvision
from torch.utils.cpp_extension import CUDA_HOME

print("Python:", sys.version)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA 兼容版本:", torch.version.cuda, "SDK:", CUDA_HOME)
assert torch.cuda.is_available(), "当前进程无法访问 PPU，请先检查设备分配及运行权限"
print("可见卡数:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    x = torch.ones((2, 2), device=f"cuda:{i}", requires_grad=True)
    (x @ x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4))
print("基础计算通过")
PY
```

本实例预期可见 2 张卡。如果卡数为 0，先排查 DSW 资源分配、容器/沙箱设备访问权限及平台设置的可见设备变量。
受限沙箱曾无法检测到设备，而同一实例的正常执行环境可以访问两张卡；不要据此重装 PyTorch。
后续命令若失败，应先解决该步骤的问题再继续。

### 2. 创建继承镜像依赖的虚拟环境

使用 `--system-site-packages` 继承已经适配 PPU 的 PyTorch，新增依赖安装到项目虚拟环境。
以下目录首次使用时应不存在；如果已有同名环境，请先确认其用途。

```bash
cd /mnt/workspace/omni3d
python -m venv --system-site-packages /mnt/workspace/venvs/cubercnn-ppu
source /mnt/workspace/venvs/cubercnn-ppu/bin/activate

# 保存实际版本约束，防止依赖解析替换镜像的核心包。
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

names = ("torch", "torchvision", "torchaudio", "numpy", "Pillow")
p = Path(sys.prefix) / "ppu-constraints.txt"
p.write_text("".join(f"{name}=={version(name)}\n" for name in names))
print("约束文件:", p)
PY
export PIP_CONSTRAINT="$VIRTUAL_ENV/ppu-constraints.txt"
export MPLBACKEND=Agg
```

新开终端时重新执行 `source` 和上述两个 `export`。安装命令使用 `python -m pip`，确保对应当前虚拟环境。
沿用平台配置的包源；如需获取 PPU 专用包，向 PAI 支持确认适配此 SDK 的包及访问方式，不要直接替换成普通 CUDA wheel。

### 3. 安装通用依赖

以下版本作为 Python 3.8 环境的安装起点；Detectron2 的依赖也显式列出，便于后面用 `--no-deps` 安装编译扩展。
使用无图形界面的 OpenCV 版本适配无桌面的 DSW，避免同时安装多个 OpenCV Python 发行包。

```bash
python -m pip install \
  "setuptools==68.2.2" "wheel==0.40.0" "ninja==1.10.0" "Cython==0.29.36"
python -m pip install \
  "fvcore==0.1.5.post20221221" "iopath==0.1.9" \
  "opencv-python-headless==4.8.1.78" "pycocotools==2.0.7" \
  "scipy==1.9.3" "pandas==1.5.3" "matplotlib==3.7.5" "seaborn==0.12.2" \
  "yacs==0.1.8" "termcolor==2.4.0" "tabulate==0.9.0" \
  "cloudpickle==2.2.1" "tqdm==4.64.0" "tensorboard==2.13.0" \
  "omegaconf==2.3.0" "hydra-core==1.3.2" "black==24.8.0" "packaging>=22.0"
```

### 4. 安装 Detectron2 和 PyTorch3D 的编译扩展

如果 PAI 提供匹配 **Python 3.8、当前 PyTorch 和 PPU SDK 2.1.0** 的这两个包，优先按其指定版本安装。
尚未确认此镜像对应的专用 wheel，因此这里提供源码编译备选路径，而不虚构 PPU 包版本或下载地址。

[Detectron2 官方安装说明](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)支持从源码构建。
[PyTorch3D v0.7.4 安装说明](https://github.com/facebookresearch/pytorch3d/blob/v0.7.4/INSTALL.md)
列出了 Python 3.8 和 PyTorch 2.0.0/2.0.1；这只是选择候选版本的依据，不等于已验证本镜像的 `2.0.0a0` 定制版本。
Detectron2 使用主分支作为待验证候选，安装后记录提交号；正式复现应固定到实际验证通过的提交。

```bash
# 先确认第 1 步通过，且当前已激活 cubercnn-ppu 虚拟环境。
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
command -v gcc
command -v g++
"$CUDA_HOME/bin/nvcc" --version

# 在本机 SDK 上构建；不使用原 README 中 cu101/torch1.8 的预编译 wheel。
# --no-build-isolation 让编译使用当前 PPU PyTorch；--no-deps 避免替换预装框架。
mkdir -p /mnt/workspace/omni3d-deps
git clone https://github.com/facebookresearch/detectron2.git /mnt/workspace/omni3d-deps/detectron2
git clone --branch v0.7.4 --depth 1 https://github.com/facebookresearch/pytorch3d.git /mnt/workspace/omni3d-deps/pytorch3d

FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps /mnt/workspace/omni3d-deps/detectron2
FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps /mnt/workspace/omni3d-deps/pytorch3d

git -C /mnt/workspace/omni3d-deps/detectron2 rev-parse HEAD
git -C /mnt/workspace/omni3d-deps/pytorch3d rev-parse HEAD
python -m pip check
```

已有源码目录时复用并核对其版本，不要重复克隆或直接覆盖。`FORCE_CUDA=1` 仅要求构建加速扩展，不能修复设备不可见或不支持的算子。
保留镜像的架构配置，不要照抄 NVIDIA 卡的 `TORCH_CUDA_ARCH_LIST`。
如果出现编译错误、`undefined symbol` 或不支持的算子，保存首个错误及版本信息，向 PAI 支持确认适配方式；不要通过升级普通 PyTorch 解决。
`pip check` 也可能报告镜像已有包的冲突，应逐项判断，不要直接批量升级。

### 5. 检查算子及项目导入

仅能 `import` 不足以说明加速扩展可用。下面检查两张卡上的 NMS、ROIAlign 前反向和 PyTorch3D 3D IoU，并导入项目模块：

```bash
cd /mnt/workspace/omni3d
python - <<'PY'
import torch
from detectron2 import _C as detectron2_C
from detectron2.layers import ROIAlign, nms
from pytorch3d.ops import box3d_overlap
from cubercnn import util, vis
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.evaluation import Omni3DEvaluationHelper

assert torch.cuda.is_available(), "PPU 不可用"
for i in range(torch.cuda.device_count()):
    device = f"cuda:{i}"
    boxes = torch.tensor([[0., 0., 4., 4.], [1., 1., 3., 3.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert nms(boxes, scores, 0.5).cpu().tolist() == [0, 1]
    x = torch.ones((1, 2, 8, 8), device=device, requires_grad=True)
    rois = torch.cat([torch.zeros((2, 1), device=device), boxes], dim=1)
    y = ROIAlign((2, 2), spatial_scale=1.0, sampling_ratio=0, aligned=True)(x, rois)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    cube = torch.tensor([[
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]], dtype=torch.float32, device=device)
    _, iou = box3d_overlap(cube, cube)
    torch.testing.assert_close(iou, torch.ones_like(iou))
    print(device, torch.cuda.get_device_name(i), "算子检查通过")
print("项目模块导入通过；仍需实际演示验证模型和渲染路径")
PY

python demo/demo.py --help
python tools/train_net.py --help
```

### 6. 先运行单卡演示

依赖检查通过后，运行下方[演示](#demo)的示例命令。DSW 使用上述无图形界面的 OpenCV 版本时，**去掉 `--display`**，
保留 `OUTPUT_DIR output/demo`，从输出目录查看结果。示例图片、模型权重和配置仍需能从原下载地址获取。
完成一次有检测结果的演示，才能进一步验证模型前向及 PyTorch3D 渲染路径。

训练还需要按 [DATA.md](DATA.md) 准备数据。先用单卡确认训练，再尝试 `--num-gpus 2`；
双卡通信和训练尚未验证，通信配置需遵循 PAI 对当前镜像的要求。下方训练配置原本面向 48 卡，不能直接套用到两张 PPU，
需按 [超参数调整说明](#tuning) 调整批量大小、学习率及迭代次数。

## 演示 <a name="demo"></a>

使用在完整 Omni3D 数据集上训练的 `DLA34` 模型，对指定文件夹中的图像运行 Cube R-CNN 演示：

``` bash
# 下载 COCO 示例图像
sh demo/download_demo_COCO_images.sh

# 运行演示示例
python demo/demo.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-folder "datasets/coco_examples" \
--threshold 0.25 --display \
MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
OUTPUT_DIR output/demo 
```

更多细节请参阅 [`demo.py`](demo/demo.py)。例如，如果已知相机内参，可通过 `--focal-length <float>` 和 `--principal-point <float> <float>` 传入焦距及主点坐标，其中 `<float>` 表示浮点数。更多模型检查点请参阅 [`MODEL_ZOO.md`](MODEL_ZOO.md)。

## Omni3D 数据 <a name="data"></a>

请参阅 [`DATA.md`](DATA.md)，了解如何下载并配置 Omni3D 基准的图像和标注，用于训练与评估 Cube R-CNN。

## 在 Omni3D 上训练 Cube R-CNN <a name="training"></a>

项目提供了在以下数据集上训练 Cube R-CNN 的配置文件：

* Omni3D: [`configs/Base_Omni3D.yaml`](configs/Base_Omni3D.yaml)
* Omni3D 室内数据：[`configs/Base_Omni3D_in.yaml`](configs/Base_Omni3D_in.yaml)
* Omni3D 室外数据：[`configs/Base_Omni3D_out.yaml`](configs/Base_Omni3D_out.yaml)

原作者使用 48 张 GPU 训练，并通过 [submitit](https://github.com/facebookincubator/submitit) 封装以下训练命令：

```bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml \
  OUTPUT_DIR output/omni3d_example_run
```

注意，提供的配置使用针对 48 张 GPU 调整过的超参数。也可以使用以下命令在单张 GPU 上训练，但无法保证达到原论文的最终性能：

``` bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025 \
  SOLVER.MAX_ITER 5568000 SOLVER.STEPS (3340800, 4454400) \
  SOLVER.WARMUP_ITERS 174000 TEST.EVAL_PERIOD 1392000 \
  VIS_PERIOD 111360 OUTPUT_DIR output/omni3d_example_run
```

### 超参数调整建议 <a name="tuning"></a>

Omni3D 配置面向多节点训练设计。

原作者采用简单的缩放规则来适配不同的硬件配置。使用 `DLA34` 骨干网络训练时，16GB 显存的 GPU（例如 V100）每卡每批可容纳 4 张图像。设 GPU 数量为 $g$，则每批图像总数为 $b = 4g$。定义推荐批量大小 $b_0$ 与实际批量大小 $b$ 的比值为 $r = b_0 / b$。$b_0$ 的值可在配置文件中找到，例如完整 Omni3D 训练使用 $b_0 = 196$，见[对应配置](https://github.com/facebookresearch/omni3d/blob/main/configs/Base_Omni3D.yaml#L4)。
按以下规则缩放超参数：

  * `SOLVER.IMS_PER_BATCH` $=b$
  * `SOLVER.BASE_LR` $/=r$
  * `SOLVER.MAX_ITER`  $*=r$
  * `SOLVER.STEPS`  $*=r$
  * `SOLVER.WARMUP_ITERS` $*=r$
  * `TEST.EVAL_PERIOD` $*=r$
  * `VIS_PERIOD`  $*=r$

原作者调整 GPU 数量 $g$，使 `SOLVER.MAX_ITER` 大致处于 9 万至 12 万次迭代之间。不同 GPU 配置的性能不一定相同；在资源配置差异较大时（例如仅使用单张 GPU），模型性能可能出现明显差异。

## 在 Omni3D 上进行推理 <a name="inference"></a>

运行以下命令，评估 Cube R-CNN [`MODEL_ZOO.md`](MODEL_ZOO.md) 中提供的已训练模型：

```
python tools/train_net.py \
  --eval-only --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
  MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
  OUTPUT_DIR output/evaluation
```

评估方式类似于 COCO，采用 $IoU_{3D}$（由 [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py) 实现）作为指标，并对各类别取平均，计算整体三维检测性能。

如果需要在 Cube R-CNN 评估流程之外评估自己的模型，建议使用[评估模块](https://github.com/facebookresearch/omni3d/blob/main/cubercnn/evaluation/omni3d_evaluation.py#L60-L88)中的 `Omni3DEvaluationHelper` 类，用法可参考[训练脚本中的示例](https://github.com/facebookresearch/omni3d/blob/main/tools/train_net.py#L68-L114)。

评估器通过 Detectron2 的 `MetadataCatalog` 记录类别名称及连续编号，因此需要正确设置以下变量：

```
# (list[str]) 按连续类别编号顺序排列的类别名称
MetadataCatalog.get('omni3d_model').thing_classes = ... 

# (dict[int: int]) 从 Omni3D 原始类别编号到连续类别编号的映射
MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = ...
```

评估器接收按图像组织的预测结果列表，其中每张图像的结果格式如下（类型和中文说明为占位说明）：

```
{
    "image_id": <int> Omni3D 中图像的唯一标识符,
    "K": <np.array> 图像对应的 3×3 相机内参矩阵,
    "width": <int> 图像宽度,
    "height": <int> 图像高度,
    "instances": [
        {
            "image_id": <int> Omni3D 中图像的唯一标识符,
            "category_id": <int> 预测类别的连续编号,
                可通过以下映射将 Omni3D 原始类别编号转换为连续编号：
                MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
            "bbox": [float] 用于计算 IoU2D 的二维框，格式为 [x1, y1, x2, y2],
            "score": <float> 目标的置信度分数,
            "depth": <float> 目标中心的深度,
            "bbox3D": list[list[float]] 用于计算 IoU3D 的 8×3 角点坐标,
        }
        ...
    ]
}
```

## 实验结果 <a name="results"></a>

Cube R-CNN 的详细性能及与其他方法的比较，请参阅 [`RESULTS.md`](RESULTS.md)。

## 许可证 <a name="license"></a>

Cube R-CNN 根据 [CC-BY-NC 4.0](LICENSE.md) 许可证发布。

## 引用 <a name="citing"></a>

如果在研究中使用 Omni3D、Cube R-CNN 或引用本项目的结果，请使用以下 BibTeX 条目。为保证文献引用准确，条目中的论文题名、作者等信息保留原文。

```BibTeX
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```

如果使用 Omni3D 基准，也请引用其包含的所有数据集。对应的 BibTeX 条目如下。

<details><summary>数据集引用（BibTeX）</summary>

```BibTex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}
``` 

```BibTex
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```

```BibTex
@inproceedings{song2015sun,
  title={Sun rgb-d: A rgb-d scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}
```

```BibTex
@inproceedings{dehghan2021arkitscenes,
  title={{ARK}itScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB}-D Data},
  author={Gilad Baruch and Zhuoyuan Chen and Afshin Dehghan and Tal Dimry and Yuri Feigin and Peter Fu and Thomas Gebauer and Brandon Joffe and Daniel Kurz and Arik Schwartz and Elad Shulman},
  booktitle={NeurIPS Datasets and Benchmarks Track (Round 1)},
  year={2021},
}
```

```BibTex
@inproceedings{hypersim,
  author    = {Mike Roberts AND Jason Ramapuram AND Anurag Ranjan AND Atulit Kumar AND
                 Miguel Angel Bautista AND Nathan Paczan AND Russ Webb AND Joshua M. Susskind},
  title     = {{Hypersim}: {A} Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
  booktitle = {ICCV},
  year      = {2021},
}
```

```BibTex
@article{objectron2021,
  title={Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations},
  author={Ahmadyan, Adel and Zhang, Liangkai and Ablavatski, Artsiom and Wei, Jianing and Grundmann, Matthias},
  journal={CVPR},
  year={2021},
}
```

</details>

[gg]: https://github.com/gkioxari
[jj]: https://github.com/jcjohnson
[gb]: https://github.com/garrickbrazil
[ak]: https://github.com/abhi1kumar
[nr]: https://github.com/nikhilaravi
[js]: https://github.com/jstraub
[d2]: https://github.com/facebookresearch/detectron2
[py3d]: https://github.com/facebookresearch/pytorch3d
[pyt]: https://pytorch.org/
[coco]: https://cocodataset.org/
