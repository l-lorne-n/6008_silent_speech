from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from train_auto_segment_cnn import boundaries, augmented_batch, cnn, np, torch


def test_boundary_augmentation_stays_inside_trial_and_input_capacity():
    for start,end,hi in [(0,400,400),(200,600,800),(200,6200,6400),(0,6144,6144)]:
        for ds in [-200,0,200]:
            for de in [-200,0,200]:
                a,b=boundaries(start,end,0,hi,ds,de)
                assert 0<=a<b<=hi and 400<=b-a<=6144


def test_normalization_does_not_fit_validation_or_padding():
    raw=np.zeros((3,2,6144),np.float32)
    raw[0,:,:400]=2;raw[1,:,:800]=4;raw[2,:,:600]=100000
    x,norm=cnn.normalize(raw,np.array([400,800,600]),np.array([0,1]))
    np.testing.assert_allclose(norm['mean'],[10/3,10/3])
    assert torch.count_nonzero(x[0,:,400:])==0
    assert norm['fit_trials']==2


def test_augmented_batch_never_uses_context_padding():
    contexts=np.full((1,2,6544),1e8,dtype=np.float32)
    contexts[:,:,:1000]=3
    data=dict(contexts=contexts,lengths=np.array([600]),offsets=np.array([200]),context_lengths=np.array([1000]))
    cnn.set_seed(20261021)
    for _ in range(20):
        x=augmented_batch(data,np.array([0]),dict(mean=[0.,0.],scale=[1.,1.]))
        assert x.shape==(1,2,6144) and torch.isfinite(x).all()
        assert float(x.max())<=3.31
        assert torch.count_nonzero(x[:,:,1200:])==0
