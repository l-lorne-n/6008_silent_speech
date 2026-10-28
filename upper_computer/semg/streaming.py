"""Experimental event-based classifier. Detector never sees target/cue metadata."""
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from .activity import ActivityConfig, ActivityDetector
from .lda import extract_recording
from .channels import validate_channels, recording_channels, require_channels

SCHEMA='semg_activity_lda_v2'
FEATURE_VERSION='continuous_notch50_padded_segment_quarters_v1'


def validate_model(m):
    if m.get('schema') not in (SCHEMA,'semg_activity_lda_v1') or m.get('features')!=FEATURE_VERSION:
        raise ValueError('需要活动切段模型；旧版按提示切段的LDA模型不能直接用于此窗口')
    ActivityConfig(**m['detector'])
    if m['schema']==SCHEMA and 'channels' not in m:raise ValueError('活动模型缺少通道配置')
    channels=validate_channels(m.get('channels'))
    if m['schema']=='semg_activity_lda_v1' and channels!=[0]:raise ValueError('旧活动模型只支持CH0')
    dimensions=8*len(channels)
    classes=m['classes'];n=len(classes)
    if n<2 or len(set(classes))!=n or not all(isinstance(w,str) and w for w in classes):
        raise ValueError('模型词表无效')
    for key,shape in [('mean',(dimensions,)),('scale',(dimensions,)),('coef',(1 if n==2 else n,dimensions)),('intercept',(1 if n==2 else n,))]:
        a=np.asarray(m[key],float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('模型参数无效：'+key)
    if np.any(np.asarray(m['scale'])<=0):raise ValueError('模型标准差无效')
    if not (0<=m['min_probability']<=1 and 0<=m['min_margin']<=1):raise ValueError('拒识参数无效')
    return m


def load_model(path):
    if Path(path).stat().st_size>2_000_000:raise ValueError('模型文件过大')
    return validate_model(json.loads(Path(path).read_text(encoding='utf-8')))


def classify(model,features):
    z=(np.asarray(features)-model['mean'])/model['scale']
    scores=np.asarray(model['coef'])@z+model['intercept']
    if not np.isfinite(scores).all():raise ValueError('模型输出非有限值')
    if len(model['classes'])==2:scores=np.array([0.,scores[0]])
    p=np.exp(scores-scores.max());p/=p.sum();order=np.argsort(p)
    best=int(order[-1]);prob=float(p[best]);margin=prob-float(p[order[-2]])
    accepted=prob>=model['min_probability'] and margin>=model['min_margin']
    return dict(word=model['classes'][best] if accepted else None,candidate=model['classes'][best],
                probability=prob,margin=margin,accepted=accepted,
                reason='accepted' if accepted else 'low_confidence')


class Recognizer:
    def __init__(self,model):
        self.model=validate_model(model)
        self.channels=validate_channels(model.get('channels'))
        self.detector=ActivityDetector(ActivityConfig(**model['detector']),len(self.channels))

    def feed(self,values,first):
        result=[]
        for e in self.detector.feed(values,first):
            if e['kind']=='segment':e=dict(e,**classify(self.model,e['features']),kind='prediction')
            result.append(e)
        return result


def recording_stream(folder,channels=None):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
    if meta.get('sampling_rate_hz') != 1000:
        raise ValueError('旧活动检测/在线模型仅支持1000 Hz；新版2000 Hz数据请使用离线20–500 Hz LDA/CNN流程')
    channels=recording_channels(meta) if channels is None else validate_channels(channels)
    require_channels(meta,channels)
    with (folder/'samples.csv').open(encoding='utf-8-sig',newline='') as f:
        data=np.array([(int(r['device_sample_idx']),*[int(r[f'raw_ch{c}']) for c in channels]) for r in csv.DictReader(f)],dtype=np.int64)
    if not len(data) or np.any(np.diff(data[:,0])!=1):raise ValueError('录制为空或存在缺样')
    return data


def detect_recording(data,config):
    detector=ActivityDetector(config,data.shape[1]-1);events=[]
    for start in range(0,len(data),1000):
        part=data[start:start+1000];events.extend(detector.feed(part[:,1:],int(part[0,0])))
    return events,detector.status()


def assign_labels(events,rows):
    """Cue intervals used only after label-free detection, for training/scoring."""
    mapping={};ambiguous=[]
    for i,e in enumerate(events):
        if e['kind']!='segment':continue
        matches=[r for r in rows if max(0,min(e['end'],r['rest'])-max(e['start'],r['action'])) >= .5*(e['end']-e['start'])]
        if len(matches)==1:mapping[i]=matches[0]
        elif matches:ambiguous.append(i)
    counts={r['trial_id']:sum(v['trial_id']==r['trial_id'] for v in mapping.values()) for r in rows}
    # Multiple candidates for one prompted word are not multiple independent labels.
    unique={i:r for i,r in mapping.items() if counts[r['trial_id']]==1}
    return unique,dict(ambiguous_events=ambiguous,multiple_candidate_trial_ids=[k for k,v in counts.items() if v>1])


def train_activity(folder,output_root,progress=lambda _:None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    folder=Path(folder)
    channels=recording_channels(json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig')))
    progress('核查录制并连续检测活动；检测器不读取提示词或动作标记…')
    rows,excluded,source=extract_recording({'path':str(folder)},'notch50',channels=channels)
    if len(rows)<10:raise ValueError('完整试次不足，请先采集多轮词语')
    config=ActivityConfig();data=recording_stream(folder)
    if min(r['prepare'] for r in rows)-int(data[0,0])<config.calibration_ms:
        raise ValueError('开头需至少5秒连续静息校准；请使用含静息基线的录制')
    blocks=sorted({r['block'] for r in rows})
    if len(blocks)<3:raise ValueError('至少需要3整轮；前80%轮次训练、最后20%轮次留出')
    split=max(1,int(len(blocks)*.8));train_blocks=blocks[:split];test_blocks=blocks[split:]
    cutoff=min(r['prepare'] for r in rows if r['block'] in test_blocks)
    events,last_status=detect_recording(data,config);labels,label_notes=assign_labels(events,rows)
    train=[i for i,r in labels.items() if r['block'] in train_blocks and events[i]['emitted']<cutoff]
    classes=sorted({r['word'] for r in rows})
    counts={w:sum(labels[i]['word']==w for i in train) for w in classes}
    if min(counts.values())<2:raise ValueError('自动切段后每词训练样本至少2条，当前：'+str(counts))
    x=np.array([events[i]['features'] for i in train]);y=np.array([classes.index(labels[i]['word']) for i in train])
    progress('按完整轮次划分，拟合活动片段分类器…')
    scaler=StandardScaler().fit(x);est=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(x),y)
    model=dict(schema=SCHEMA,features=FEATURE_VERSION,detector=asdict(config),classes=classes,channels=channels,
               mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=est.coef_.tolist(),intercept=est.intercept_.tolist(),
               min_probability=.6,min_margin=.2,experimental=True,
               rejection_note='默认启发式阈值，未经独立非指令数据校准；不是可靠的未知词检测',
               source=source,train_blocks=train_blocks,test_blocks=test_blocks,
               training_trial_ids=[labels[i]['trial_id'] for i in train])
    validate_model(model)
    # Verify portable binary/multiclass probability implementation against sklearn.
    for features,prob in zip(x,est.predict_proba(scaler.transform(x))):
        p=classify(model,features)
        if not np.isclose(p['probability'],prob.max()):raise ValueError('导出模型与训练预测不一致')
    test_rows=[r for r in rows if r['block'] in test_blocks]
    matches={labels[i]['trial_id']:i for i in labels if labels[i]['block'] in test_blocks and events[i]['start']>=cutoff}
    results=[]
    for r in test_rows:
        i=matches.get(r['trial_id']);p=classify(model,events[i]['features']) if i is not None else None
        results.append(dict(trial_id=r['trial_id'],word=r['word'],detected=i is not None,prediction=p,
                            correct=bool(p and p['word']==r['word'])))
    extras=[]
    for i,e in enumerate(events):
        if e['kind']=='segment' and e['start']>=cutoff and i not in matches.values():
            extras.append(dict(e,**classify(model,e['features'])))
    # Verify all input snapshots used for extraction remained unchanged.
    for name,digest in source['hashes'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('源文件在分析时发生变化')
    report=dict(scope='实验性：同次录制，前80%完整轮训练、后20%留出；不是跨批次或真人在线验证',channels=channels,
                source=source,config=asdict(config),train_counts=counts,train_blocks=train_blocks,test_blocks=test_blocks,
                excluded_by_existing_checks=excluded,label_notes=label_notes,total_segments=sum(e['kind']=='segment' for e in events),
                test_trials=len(results),test_detected=sum(r['detected'] for r in results),test_correct=sum(r['correct'] for r in results),
                test_results=results,unmatched_test_candidates=extras,
                unmatched_test_accepted=sum(e['accepted'] for e in extras),
                test_duration_s=(int(data[-1,0])+1-cutoff)/1000,
                final_detector_status=last_status,
                caveat='未匹配候选包括休息活动、边界错位和重复切段，不能全部断定为非指令；拒识阈值未经校准。检测质量与分类质量分别报告。')
    out=Path(output_root)/('activity_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    for name,obj in [('model.json',model),('report.json',report),('detection_events.json',events)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return dict(model_path=str(out/'model.json'),report=report)


def replay_activity(model_path,folder,output_root,progress=lambda _:None):
    model=load_model(model_path);data=recording_stream(folder,model.get('channels',[0]));engine=Recognizer(model);events=[]
    progress('连续回放中，识别器仅接收原始数据及样本序号…')
    for start in range(0,len(data),1000):
        part=data[start:start+1000];events.extend(engine.feed(part[:,1:],int(part[0,0])))
    digest=hashlib.sha256((Path(folder)/'samples.csv').read_bytes()).hexdigest()
    overlaps=digest==model['source']['hashes']['samples.csv']
    result=dict(source=str(folder),includes_training_recording=overlaps,events=events,final_status=engine.detector.status(),
                note='整段回放不是独立准确率；此处未使用标签评分。' if not overlaps else '本次回放包含训练来源，不能当作独立效果。')
    out=Path(output_root)/('activity_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(folder=str(out),report=result)
