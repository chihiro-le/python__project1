# 用 NumPy 实现 OpenCLIP 图像编码器

本项目实现课程提供的 OpenCLIP ViT-B-32 图像编码器的前向计算。图片经过预处理后以形状 [B, 3, 224, 224] 的 float32 数组进入编码器。编码器为每张图返回一个 512 维单位向量。模型使用课程提供的固定权重。

## 项目结构

| 文件 | 用途 |
| --- | --- |
| image_encoder.py | 切块、输入 token、多头注意力、Transformer 模块和输出投影 |
| preprocess_image.py | 读取图像后缩放、中心裁剪和标准化 |
| helpers.py | LayerNorm、GELU、注意力权重和向量归一化 |
| runtime.py、config.json | 配置及模型权重、图像数据的加载 |
| check.py | 资源校验和逐阶段的数值对比 |
| run.py | 用默认图片或指定图片生成 .npz 特征文件 |
| requirements.txt | NumPy、Pillow 和 imageio 的版本范围 |
| transformer_steps.py | 课程提供的小规模 Transformer 数值演示 |

## 准备课程数据

从课程平台获取并解压原始 project1-学生版.zip。将原包中的以下内容复制到本项目的同名路径，保留本项目中的 image_encoder.py。

- weights/image_encoder.npz
- assets/reference_outputs.npz
- assets/images/ 整个目录

三处路径与原包一致，配套 JSON 清单已随本项目提供。check.py 会核对课程数据的校验信息。已有完整原包的情况下，也可以只将本项目的 image_encoder.py 复制到原包目录进行运行。

## 复现

要求 Python 3.9 或更高版本。打开包含 check.py 和 requirements.txt 的文件夹，在终端依次运行：

    python -m pip install -r requirements.txt
    python check.py --stage all
    python run.py --output output_tokens.npz

使用自己的图片时，在 --images 后依次写入图片路径：

    python run.py --images "path/to/a.jpg" "path/to/b.png" --output my_tokens.npz

输出 .npz 包含 ids 和 output_tokens。ids 与输入顺序一致，output_tokens 的形状为 [B, 512]，每行对应一张图片。

验收环境为 Windows、Python 3.14.7、NumPy 2.5.5、Pillow 12.3.0、imageio 2.38.2。老师提供的八项检查均通过。默认输出为 (10, 512) 的 float32 数组，output 阶段最大绝对误差为 5.960e-07。

## 实现说明

每张图先分成 49 个 32 × 32 的小块，每块按通道、行、列的顺序展开。投影后在序列开头加入 CLS，再加位置嵌入。12 个 Transformer 模块更新 50 个 token。末端选出 CLS 行，依次进行 LayerNorm、投影和单位向量归一化。

课程参考数组只用于 check.py 中的验证。课程框架、模型参数和测试材料由教师提供。项目实现、数值复核及运行说明由 ChatGPT 协助完成。

