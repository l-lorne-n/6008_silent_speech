from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from speechnet_two_channel import SpeechNet2, ExportModel, resources, VARIANTS, torch, nn


def test_matches_upstream_with_explicit_two_channel_configuration():
    """Ensure fixed pooling and adaptation preserve the intended upstream operator sequence."""
    upstream = Path(__file__).resolve().parents[2]/'silent_wear/SilentWear/models/cnn_architectures/SpeechNet.py'
    spec = importlib.util.spec_from_file_location('upstream_speechnet_reference', upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    for variant, temporal in VARIANTS.items():
        blocks = [dict(out_channels=c, kernel=(1, k), pool=(1, p))
                  for c, k, p in zip([8, 16, 16], temporal, [8, 4, 4])]
        blocks += [dict(out_channels=32, kernel=(2, 1), pool=(1, 1)),
                   dict(out_channels=32, kernel=(1, 1), pool=(1, 1))]
        reference = module.SpeechNet(C=2, T=6144, output_classes=10, blocks_config=blocks,
                                     p_dropout=.5, global_pool='avg').eval()
        adapted = SpeechNet2(variant).eval()
        for block_id, block in enumerate(reference.blocks):
            for layer_id in [0, 1]:
                adapted.layers[block_id*4+layer_id].load_state_dict(block[layer_id].state_dict())
        adapted.layers[-1].load_state_dict(reference.fc.state_dict())
        x = torch.randn(2, 2, 6144)
        with torch.inference_mode():
            torch.testing.assert_close(adapted(x), reference(x), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(ExportModel(adapted)(x.unsqueeze(2)), reference(x), rtol=1e-5, atol=1e-6)


def test_temporal_blocks_do_not_mix_electrodes_before_spatial_fusion():
    net = SpeechNet2().eval()
    x = torch.randn(1, 1, 2, 6144)
    changed = x.clone()
    changed[:, :, 1, :] += 100
    with torch.inference_mode():
        a = net.layers[:12](x)
        b = net.layers[:12](changed)
    torch.testing.assert_close(a[:, :, 0], b[:, :, 0], rtol=0, atol=0)
    assert not torch.equal(a[:, :, 1], b[:, :, 1])
    assert net.layers[12].kernel_size == (2, 1)


def test_resources_use_actual_shapes_and_keep_ten_classes():
    for variant in VARIANTS:
        net = SpeechNet2(variant)
        report = resources(net)
        assert report['parameters'] == sum(p.numel() for p in net.parameters())
        assert report['layers'][-1]['shape'] == [1, 10]
        assert report['int8_inventory_plus_reserve_bytes'] < 1024*1024
        assert report['st_compiler_verified'] is False
        assert sum(isinstance(m, nn.Conv2d) for m in net.modules()) == 5
