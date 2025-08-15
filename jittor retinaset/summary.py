import jittor as jt
from jittor import nn
from nets.retinanet import retinanet  # 假设这是你的Jittor模型


def count_flops(model, input_shape):
    model.eval()
    flops = 0

    # Jittor的hook方法统计FLOPs（需要手动实现）
    def count_conv_flops(m, x, y):
        # 计算卷积层的FLOPs: (2*Cin*K*K - 1)*Hout*Wout*Cout
        if isinstance(m, nn.Conv2d):
            cin = m.weight.shape[1]
            cout = m.weight.shape[0]
            kh, kw = m.weight.shape[2], m.weight.shape[3]
            h_out = y.shape[2]
            w_out = y.shape[3]
            flops += (2 * cin * kh * kw - 1) * h_out * w_out * cout

    # 注册前向hook
    handles = []
    for layer in model.modules():
        handles.append(layer.register_forward_hook(count_conv_flops))

    # 模拟前向传播
    dummy_input = jt.randn(1, 3, *input_shape)
    model(dummy_input)

    # 移除hook
    for h in handles:
        h.remove()

    return flops


if __name__ == '__main__':
    input_shape = [600, 600]
    num_classes = 80
    phi = 2

    # Jittor自动选择设备（无需手动指定）
    model = retinanet(num_classes, phi)

    # 参数量统计（与PyTorch相同）
    params = sum(p.numel() for p in model.parameters())
    print(f'# generator parameters: {params}')

    # FLOPs统计（自定义方法）
    flops = count_flops(model, input_shape)
    flops = flops * 2  # 与PyTorch一致，乘2以包含乘加操作


    # 格式化输出
    def clever_format(nums, format_str):
        if nums > 1e9:
            return f"{nums / 1e9:.3f}G"
        elif nums > 1e6:
            return f"{nums / 1e6:.3f}M"
        else:
            return f"{nums:.3f}"


    flops_str = clever_format(flops, "%.3f")
    params_str = clever_format(params, "%.3f")

    print(f'Total GFLOPS: {flops_str}')
    print(f'Total params: {params_str}')