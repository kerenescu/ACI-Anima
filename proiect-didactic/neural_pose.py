"""Inferență CPU pentru greutățile oficiale AnimatedDrawings (MIT).

Port de inferență pentru configurația MMPose din arhiva autorilor: ResNet-50,
trei deconvoluții, 17 heatmap-uri, flip test și decodare cu deplasare 0.25 px.
Nu executăm config.py/handler-ul și nu importăm MMCV, TorchServe sau torchvision.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / 'models'
WEIGHTS = MODEL_DIR / 'pose_resnet50.pt'
MANIFEST = MODEL_DIR / 'model.json'
SOURCE = 'https://github.com/facebookresearch/AnimatedDrawings/releases/tag/v0.0.1'
COCO_NAMES = ['nose','left_eye','right_eye','left_ear','right_ear',
              'left_shoulder','right_shoulder','left_elbow','right_elbow',
              'left_wrist','right_wrist','left_hip','right_hip','left_knee',
              'right_knee','left_ankle','right_ankle']
FLIP = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
_LOCK = threading.RLock()

def model_spec(choice='active'):
    candidate=MODEL_DIR/'pose_finetuned.pt';manifest=MODEL_DIR/'pose_finetuned.json'
    profile='official'
    if choice=='active':
        try:profile=json.loads((MODEL_DIR/'active_model.json').read_text(encoding='utf-8')).get('profile','official')
        except (OSError,ValueError):pass
    elif choice=='finetuned':profile='finetuned'
    if profile=='finetuned' and candidate.exists() and manifest.exists():
        data=json.loads(manifest.read_text(encoding='utf-8'));return dict(profile='finetuned',path=candidate,manifest=data,name='ResNet-50 · ajustat local pe ADD')
    if choice=='finetuned':raise ValueError('Modelul ajustat local nu este încă disponibil.')
    data=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    return dict(profile='official',path=WEIGHTS,manifest=data,name='AnimatedDrawings · ResNet-50 oficial')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def model_status():
    runtime = importlib.util.find_spec('torch') is not None
    weights = WEIGHTS.is_file() and MANIFEST.is_file()
    if runtime and weights:message='Model local disponibil pentru articulații · CPU. Masca se calculează separat.'
    elif weights:message='Greutățile sunt prezente, dar acest server folosește Python fără PyTorch. Repornește prin START_PROIECT.cmd din folderul în care ai rulat INSTALARE_MODEL.cmd.'
    elif runtime:message='PyTorch este disponibil, dar lipsesc fișierele modelului din această copie a proiectului. Rulează INSTALARE_MODEL.cmd în acest folder, apoi repornește serverul.'
    else:message='Lipsesc runtime-ul PyTorch și greutățile din această copie. Rulează INSTALARE_MODEL.cmd în acest folder, apoi repornește prin START_PROIECT.cmd.'
    selected=model_spec();candidate=(MODEL_DIR/'pose_finetuned.pt').exists() and (MODEL_DIR/'pose_finetuned.json').exists()
    return dict(available=runtime and weights, runtime=runtime, weights=weights,
                name=selected['name'],profile=selected['profile'],candidate_available=candidate,
                source=SOURCE,message=message)


def build_model():
    import torch.nn as nn

    class Bottleneck(nn.Module):
        def __init__(self, channels, width, stride=1):
            super().__init__()
            self.conv1=nn.Conv2d(channels,width,1,bias=False);self.bn1=nn.BatchNorm2d(width)
            self.conv2=nn.Conv2d(width,width,3,stride=stride,padding=1,bias=False);self.bn2=nn.BatchNorm2d(width)
            self.conv3=nn.Conv2d(width,width*4,1,bias=False);self.bn3=nn.BatchNorm2d(width*4)
            self.relu=nn.ReLU(inplace=True)
            self.downsample=(nn.Sequential(nn.Conv2d(channels,width*4,1,stride=stride,bias=False),
                                           nn.BatchNorm2d(width*4)) if stride!=1 or channels!=width*4 else None)

        def forward(self,x):
            shortcut=x if self.downsample is None else self.downsample(x)
            x=self.relu(self.bn1(self.conv1(x)));x=self.relu(self.bn2(self.conv2(x)))
            return self.relu(self.bn3(self.conv3(x))+shortcut)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
            self.bn1=nn.BatchNorm2d(64);self.relu=nn.ReLU(inplace=True)
            self.maxpool=nn.MaxPool2d(3,stride=2,padding=1)
            channels=64
            for stage,(width,count) in enumerate(zip([64,128,256,512],[3,4,6,3]),1):
                blocks=[Bottleneck(channels,width,1 if stage==1 else 2)]
                blocks.extend(Bottleneck(width*4,width) for _ in range(count-1))
                setattr(self,f'layer{stage}',nn.Sequential(*blocks));channels=width*4

        def forward(self,x):
            x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
            for i in range(1,5):x=getattr(self,f'layer{i}')(x)
            return x

    class Head(nn.Module):
        def __init__(self):
            super().__init__();layers=[]
            for channels in [2048,256,256]:
                layers.extend([nn.ConvTranspose2d(channels,256,4,stride=2,padding=1,bias=False),
                               nn.BatchNorm2d(256),nn.ReLU(inplace=True)])
            self.deconv_layers=nn.Sequential(*layers);self.final_layer=nn.Conv2d(256,17,1)

        def forward(self,x):return self.final_layer(self.deconv_layers(x))

    class PoseNet(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Backbone();self.keypoint_head=Head()
        def forward(self,x):return self.keypoint_head(self.backbone(x))
    return PoseNet()


def load_model(choice='active'):
    if not model_status()['available']:raise ValueError(model_status()['message'])
    selected=model_spec(choice)
    return _load_model_cached(str(selected['path']),selected['manifest'].get('sha256',''))

@lru_cache(maxsize=2)
def _load_model_cached(path,expected_digest):
    import torch
    digest=sha256_file(path)
    if digest!=expected_digest:raise ValueError('Fișierul modelului nu corespunde manifestului. Reinstalează modelul.')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    model=build_model()
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    return model.eval()


def prepare_input(image):
    """BBox întreg, padding 1.25, aspect 192/256; eșantionare biliniară RGB.

    Coordonatele pixelilor sunt cele OpenCV. Fracțiile se cuantizează la 1/32,
    ca INTER_LINEAR. Marginea din afara imaginii este neagră, ca în MMPose.
    """
    rgb=np.asarray(image.convert('RGB'));h,w=rgb.shape[:2]
    extent=np.array([max(w,h*.75),max(h,w/.75)],dtype=np.float32)*1.25
    origin=np.array([w/2,h/2],dtype=np.float32)-extent/2
    y,x=np.mgrid[:256,:192]
    sx=x*float(extent[0])/192+float(origin[0]);sy=y*float(extent[1])/256+float(origin[1])
    sx=np.rint(sx*32)/32;sy=np.rint(sy*32)/32
    ix=np.floor(sx).astype(int);iy=np.floor(sy).astype(int);fx=sx-ix;fy=sy-iy
    out=np.zeros((256,192,3),dtype=float)
    for dx,dy,weight in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx=ix+dx;yy=iy+dy;valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        out[valid]+=rgb[yy[valid],xx[valid]]*weight[valid,None]
    out=np.clip(np.floor(out+.5),0,255).astype('uint8')
    normalized=(out.astype('float32')/255-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
    return normalized.transpose(2,0,1).copy(),origin,extent,out


def decode_heatmaps(heatmaps,origin,extent):
    count,h,w=heatmaps.shape;flat=heatmaps.reshape(count,-1);idx=flat.argmax(axis=1)
    scores=flat[np.arange(count),idx].copy();xy=np.stack([idx%w,idx//w],axis=1).astype(float)
    for k,(x,y) in enumerate(xy.astype(int)):
        if 1<x<w-1 and 1<y<h-1:
            xy[k]+=.25*np.sign([heatmaps[k,y,x+1]-heatmaps[k,y,x-1],heatmaps[k,y+1,x]-heatmaps[k,y-1,x]])
    xy=xy*np.asarray(extent)/[w,h]+origin
    return xy,scores


def map_controls(points,scores):
    """Păstrează lanțurile anatomice; numele locale sunt după umărul din ecran.

    Nu sortăm fiecare cot/mână separat: un braț încrucișat își păstrează lanțul.
    Bazinul/toracele sunt mijloace; controlul capului este nasul prezis.
    """
    p=np.asarray(points);s=np.asarray(scores)
    left_arm,right_arm=([5,7,9],[6,8,10]) if p[5,0]<=p[6,0] else ([6,8,10],[5,7,9])
    # Aceeași identitate pentru braț și picior; evită răsucirea trunchiului.
    left_leg,right_leg=([11,13,15],[12,14,16]) if left_arm[0]==5 else ([12,14,16],[11,13,15])
    order=left_arm+right_arm+left_leg+right_leg
    controls=np.vstack([(p[11]+p[12])/2,(p[5]+p[6])/2,p[0],p[order]])
    quality=np.r_[min(s[11],s[12]),min(s[5],s[6]),s[0],s[order]]
    return controls,quality,order


def predict_pose(image,choice='active'):
    import torch
    started=time.perf_counter()
    with _LOCK:
        model=load_model(choice);selected=model_spec(choice);a,origin,extent,preview=prepare_input(image)
        x=torch.from_numpy(a).unsqueeze(0)
        with torch.inference_mode():
            direct=model(x)[0].numpy()
            flipped=model(x.flip(-1))[0].numpy()[FLIP,:,::-1].copy()
        shifted=flipped.copy();shifted[:,:,1:]=flipped[:,:,:-1]
        heatmaps=(direct+shifted)*.5
    if heatmaps.shape!=(17,64,48) or not np.isfinite(heatmaps).all():
        raise ValueError('Modelul a întors heatmap-uri invalide.')
    points,scores=decode_heatmaps(heatmaps,origin,extent)
    controls,quality,order=map_controls(points,scores)
    h,w=image.height,image.width;notes=[]
    outside=(controls[:,0]<0)|(controls[:,0]>=w)|(controls[:,1]<0)|(controls[:,1]>=h)
    for idx in np.flatnonzero(outside):notes.append(f'Reperul {idx+1}: predicție în afara decupajului; verifică poziția.')
    controls[:,0]=np.clip(controls[:,0],0,w-1);controls[:,1]=np.clip(controls[:,1],0,h-1)
    torso_length=float(np.linalg.norm(controls[1]-controls[0]))
    if torso_length<.06*h:notes.append('Trunchi foarte scurt sau pliat: configurație ambiguă. Verifică umerii și șoldurile.')
    for a,b,c in [(3,4,5),(6,7,8),(9,10,11),(12,13,14)]:
        lengths=np.linalg.norm(np.diff(controls[[a,b,c]],axis=0),axis=1)
        if lengths.min()<max(2,.01*h) or lengths.max()>5*max(lengths.min(),1):
            notes.append(f'Lanțul {a+1}–{b+1}–{c+1}: segmente dezechilibrate; verifică articulațiile.')
    low=np.flatnonzero(quality<.35).tolist()
    if low:notes.append('Scor neuronal mic la reperele '+', '.join(str(i+1) for i in low)+'. Verifică-le înainte de animație.')
    return dict(joints=np.round(controls,3).tolist(),quality=quality.round(4).tolist(),
                keypoints=np.c_[points,scores].round(4).tolist(),names=COCO_NAMES,
                control_order=order,heatmaps=heatmaps,origin=origin,extent=extent,
                input_image=Image.fromarray(preview),notes=notes,
                seconds=round(time.perf_counter()-started,4),source=SOURCE,
                model_name=selected['name'],model_id=selected['manifest'].get('sha256'),model_profile=selected['profile'],
                quality_meaning='Amplitudinea heatmap-ului neuronal; scor necalibrat, nu probabilitate de corectitudine')


def heatmap_overlay(image,result,control=4):
    """Heatmap reproiectat în coordonatele desenului pentru un control selectat."""
    order=result['control_order']
    if control==0:hm=(result['heatmaps'][11]+result['heatmaps'][12])*.5
    elif control==1:hm=(result['heatmaps'][5]+result['heatmaps'][6])*.5
    else:hm=result['heatmaps'][0 if control==2 else order[control-3]]
    h,w=hm.shape
    # Pillow transformă coordonate cu centre de pixel la .5.
    scale=np.array([w,h])/result['extent'];offset=-result['origin']*scale
    coeff=(float(scale[0]),0,float(offset[0]+.5-.5*scale[0]),0,float(scale[1]),float(offset[1]+.5-.5*scale[1]))
    layer=Image.fromarray(hm.astype('float32')).transform(image.size,Image.Transform.AFFINE,coeff,Image.Resampling.BILINEAR)
    a=np.asarray(layer);a=np.clip(a/max(float(hm.max()),1.e-6),0,1)
    alpha=(a*.78)[:,:,None];rgb=np.asarray(image.convert('RGB')).astype(float)
    rgb=rgb*(1-alpha)+np.array([240,97,46])*alpha
    return Image.fromarray(np.clip(rgb,0,255).astype('uint8'))
