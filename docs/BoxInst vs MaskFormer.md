
## Data Augmentation

|            BoxInst             |              Mask2Former               |
| :----------------------------: | :------------------------------------: |
| ResizeShortestEdge; RandomFlip | RandomFlip; ResizeScale; FixedSizeCrop |

- **导致的结果**: Mask2Former 随机裁切使某些样本仅存在背景，导致弱监督损失报错；
- **处理方式**: 检测到 len(instances) == 0 是，返回 dummy_loss

## Loss

### BBox Loss

- 仿照 DETR，引入 BBox 预测分支和对应的 L1 Loss, GIOU Loss
- 替换 分割分支的全监督损失为 BoxInst 中的弱监督损失

## 模型训练
#### 训练慢
> **现象**: 标准设置 (8 GPUs，batch_size=16) 下，训练需要 27 天, CPU 利用率超过 90%, 而 GPU 平均利用率仅为 25%;

> **原因**:
> - 使用 Dockerfile 构建 Docker 时，编译 mask2former/modeling/pixel_decoder/ops 中gpu 代码失效；
> - LAB 空间计算图像计算颜色相似度时, skimage.color.rgb2lab 函数只能在 CPU 上运行, 导致频繁GPU 与 CPU 频繁进行数据交换；
> -  
- **方案**: Dockfile 中配置环境变量
    ```Dockerfile
    ENV FORCE_CUDA="1"
    ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
    ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
    ```
- **效果**: GPU 平均利用率提升至 70%, 训练所需时长缩短至 2 天。
