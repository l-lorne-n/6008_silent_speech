from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_sliding_rest as run


def test_normalization_ignores_heldout_waveform():
    np=run.np
    records=[dict(recording='r',first=0,annotations=[
        dict(prepare=0,context_end=10,split='train'),
        dict(prepare=10,context_end=20,split='validation'),
        dict(prepare=20,context_end=30,split='test')])]
    waves={'r':np.vstack([np.full((10,2),2.),np.full((10,2),4.),np.full((10,2),1e10)])}
    assert run.normalization(records,waves,{'train'})['mean']==[2.,2.]
    assert run.normalization(records,waves,{'train','validation'})['mean']==[3.,3.]


def test_one_balanced_synthetic_training_batch_and_checkpoint_roundtrip(monkeypatch):
    np,torch=run.np,run.torch
    torch.set_num_threads(2)
    monkeypatch.setattr(run,'PER_CLASS',2)
    run.cnn.set_seed(123)
    length=1600
    waves={'r':np.random.default_rng(42).normal(size=(length*22,2)).astype(np.float32)}
    rows=[dict(recording='r',subject='s',trial_id=i,label=i%11,offset=i*length,
               start=i*length,end=(i+1)*length,fingerprint=str(i)) for i in range(22)]
    ix=np.arange(22);norm=dict(mean=[0.,0.],scale=[1.,1.])
    net=run.SlidingNet(800)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    loss,acc=run.train_epoch(net,optimizer,rows,ix,waves,norm,42)
    assert np.isfinite(loss) and 0<=acc<=1
    cp=run.cp_for(net,norm,{},list(range(11)),rows,ix)
    other=run.restore(cp).eval();net.eval()
    x=torch.from_numpy(run.batch(rows,ix[:2],waves,norm))
    with torch.inference_mode():torch.testing.assert_close(net(x),other(x),rtol=0,atol=0)


def test_validation_calibration_selection_cannot_draw_from_test_indices():
    rows=[dict(label=c,subject='s',recording='r',trial_id=t,split=s)
          for s in ['train','validation','test'] for c in range(11) for t in range(4)]
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    picked=run.balanced_indices(rows,train,50,42,False)
    assert len(set(picked))==len(picked)
    assert all(rows[i]['split']=='train' for i in picked)
