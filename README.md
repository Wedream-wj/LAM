# LAM

## 参考代码

代码链接：https://github.com/Fried-Rice-Lab/FriedRiceLab

## 代码结构

checkpoints：放置（超分、去雾）模型权重

demo：用来做测试的图片

interpretation：输出的归因图像文件夹路径

models：实现模型的加载方法

utils：工具类，实现归因分析方法，代码主要来自https://github.com/Fried-Rice-Lab/FriedRiceLab

interpret.py：主函数入口

## 使用方法

```bash
python interpret.py --img_path "demo/Urban7/7.png" --patch_x 110 --patch_y 150 \
--window_size 16 --output_dir interpretation

# img_path:归因HR图像路径
# patch_x:patch的x坐标
# patch_y:patch的y坐标
# window_size:patch的尺寸
# output_dir:输出文件夹
```

**注**：若想实现去雾模型，或者其他模型的归因，需要自己写get_model方法加载模型。

## 代码分析

- 该归因方法输入的是HR图像，代码会使用双三次下采样将HR图像变为LR图像，再调用超分模型将LR图像超分为SR图像，并根据梯度进行归因分析。
- 目前代码默认适用于4倍超分，若要修改，需要修改调用vis_saliency(map, zoomin=4)时的zoomin参数，需要修改调用vis_saliency_kde(map, zoomin=4)时的zoomin参数，zoomin修改为超分的倍数即可。
