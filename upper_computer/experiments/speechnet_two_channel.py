# Copyright ETH Zurich 2026 (SpeechNet architecture).
# Two-channel adaptation and resource accounting, 2026.
# SPDX-License-Identifier: Apache-2.0
# See third_party/SilentWear-LICENSE.txt and SilentWear-NOTICE.txt.
"""SpeechNet-inspired models for this project's 2 kHz, two-channel signal.

Upstream: https://github.com/pulp-bio/SilentWear
The paper's two 7x1 spatial convolutions cannot operate on two electrodes.
We replace them with 2x1 (fuse both electrodes), then 1x1 (feature mixing).
This is an adaptation, not a reproduction of the paper's 14-channel scores.
"""
import copy
from train_small_cnn import torch, nn

LENGTH = 6144
VARIANTS = {
    'paper_kernels': (4, 16, 8),
    'time_scaled': (16, 64, 32),
}


class SpeechNet2(nn.Module):
    def __init__(self, variant='time_scaled', classes=10):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        kernels = [(1, k) for k in VARIANTS[variant]] + [(2, 1), (1, 1)]
        widths = [8, 16, 16, 32, 32]
        pools = [(1, 8), (1, 4), (1, 4), (1, 1), (1, 1)]
        layers = []
        incoming = 1
        height, width = 2, LENGTH
        for channels, kernel, pool in zip(widths, kernels, pools):
            layers.extend([
                nn.Conv2d(incoming, channels, kernel, padding=(0, kernel[1] // 2)),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(pool),
            ])
            # Match upstream's symmetric integer padding, including +1 for even kernels.
            height = (height - kernel[0] + 1) // pool[0]
            width = (width + 2 * (kernel[1] // 2) - kernel[1] + 1) // pool[1]
            incoming = channels
        # Fixed equivalent of global average pooling for the fixed deployment shape.
        layers.extend([nn.AvgPool2d((height, width)), nn.Flatten(),
                       nn.Dropout(.5), nn.Linear(incoming, classes)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.unsqueeze(1))  # B, 1, electrode, time


def restore(checkpoint):
    net = SpeechNet2(checkpoint['variant'], len(checkpoint['classes']))
    net.load_state_dict(checkpoint['state_dict'])
    return net


class ExportModel(nn.Module):
    """Keep the replay API [B,2,1,6144]; network works on [B,1,2,6144]."""
    def __init__(self, net):
        super().__init__()
        self.layers = copy.deepcopy(net.layers).eval()
        for i in range(0, 20, 4):
            self.layers[i] = torch.nn.utils.fuse_conv_bn_eval(self.layers[i], self.layers[i + 1])
            self.layers[i + 1] = nn.Identity()

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


def resources(net):
    model = ExportModel(net).eval()
    x = torch.zeros(1, 1, 2, LENGTH)
    rows, macs = [], 0
    with torch.inference_mode():
        for i, layer in enumerate(model.layers):
            n = x.numel()
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                macs += x.numel() * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, nn.Linear):
                macs += layer.in_features * layer.out_features
            rows.append(dict(index=i, type=type(layer).__name__, shape=list(x.shape),
                             elements=x.numel(), adjacent_int8_bytes=n + x.numel()))
    materialized = [r for r in rows if r['type'] not in ('Identity', 'Flatten', 'Dropout')]
    inventory = 2 * LENGTH + sum(r['elements'] for r in materialized)
    weights = sum(p.numel() for n, p in model.named_parameters() if n.endswith('weight'))
    biases = sum(p.numel() for n, p in model.named_parameters() if n.endswith('bias'))
    return dict(variant=net.variant, parameters=sum(p.numel() for p in net.parameters()), macs=macs,
                int8_weights_and_int32_bias_bytes=weights + 4 * biases,
                int8_tensor_inventory_bytes=inventory, planning_reserve_bytes=256 * 1024,
                int8_inventory_plus_reserve_bytes=inventory + 256 * 1024,
                max_adjacent_int8_bytes=max(r['adjacent_int8_bytes'] for r in rows), layers=rows,
                st_compiler_verified=False, board_verified=False,
                caveat='Conservative tensor inventory plus an assumed 256KiB reserve; not an ST arena report. '
                       'Weights assumed in Flash; kernel scratch, alignment, RAM banks and latency require backend verification.')
