"""Pipeline didactic offline; rezultate reale, cu simplificări declarate.

Localizare prin contrast; segmentare Otsu + morfologie; repere editabile;
triangulare Delaunay Bowyer-Watson; ponderi de linear blend skinning.
Predictor neuronal opțional și capturi CMU ASF/AMC pentru mișcare. Fără ARAP.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
import base64
import copy
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from anatomy import (PARTS,shape_pose,structure,diagnostic_images,bind_protected,puppet_images,puppet_bridges)
from neural_pose import model_status,predict_pose,heatmap_overlay
from segmentation import choose_mask
from mocap import catalog as motion_catalog,get_clip as motion_clip

ROOT = Path(__file__).resolve().parent
VERSION = '3.8'
JOINT_NAMES = ['Bazin', 'Torace', 'Cap', 'Umăr stânga ecran', 'Cot stânga ecran',
               'Mână stânga ecran', 'Umăr dreapta ecran', 'Cot dreapta ecran',
               'Mână dreapta ecran', 'Șold stânga ecran', 'Genunchi stânga ecran',
               'Gleznă stânga ecran', 'Șold dreapta ecran', 'Genunchi dreapta ecran',
               'Gleznă dreapta ecran']
PARENTS = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]
BONES = [[p, i] for i, p in enumerate(PARENTS) if p >= 0]
EXAMPLES = [
    {'id': 'garlic_roi', 'name': 'Garlic · exemplu principal', 'note': 'Personaj izolat; repere de referință adnotate pentru demo.'},
    {'id': 'garlic_umbra', 'name': 'Garlic · umbră simulată', 'note': 'Variație controlată a aceluiași desen, nu un exemplu independent.'},
    {'id': 'garlic_contrast_redus', 'name': 'Garlic · contrast redus', 'note': 'Variație controlată: testează efectul liniilor mai slabe.'},
    {'id': 'fig6_d', 'name': 'Personaj uman · figura 6d', 'note': 'Crop din articol; verifică și ajustează reperele.'},
    {'id': 'fig6_i', 'name': 'Caz dificil · figura 6i', 'note': 'Corp și membre neobișnuite; util pentru discutarea eșecurilor.'},
    {'id': 'fig6_b', 'name': 'Caz dificil · două figuri', 'note': 'Localizarea geometrică nu poate decide semantic care este personajul.'},
]


def png_bytes(image: Image.Image) -> bytes:
    out = BytesIO(); image.save(out, format='PNG'); return out.getvalue()


def data_url(image: Image.Image) -> str:
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(image)).decode('ascii')


def decode_image(data: str, max_bytes: int = 12_000_000) -> Image.Image:
    if not isinstance(data, str) or ',' not in data:
        raise ValueError('Fișier imagine invalid.')
    raw = base64.b64decode(data.split(',', 1)[1], validate=True)
    if len(raw) > max_bytes:
        raise ValueError('Alege o imagine mai mică de 12 MB.')
    im = Image.open(BytesIO(raw))
    if im.width * im.height > 24_000_000 or min(im.size) < 8:
        raise ValueError('Imaginea trebuie să aibă între 8 pixeli pe latură și 24 megapixeli.')
    im.load(); return im


def normalize(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('RGBA')
    original=image.copy();original.info={}
    alpha=image.getchannel('A');has_alpha=alpha.getextrema()[0]<alpha.getextrema()[1]
    paper = Image.new('RGBA', image.size, 'white')
    image = Image.alpha_composite(paper, image).convert('RGB')
    w, h = image.size
    if max(w, h) / min(w, h) > 5:
        raise ValueError('Decupează imaginea în jurul unui singur personaj; raportul laturilor trebuie să fie sub 5.')
    scale = min(400 / w, 700 / h)
    size=(max(8, round(w * scale)), max(8, round(h * scale)))
    result=image.resize(size,Image.Resampling.LANCZOS);result.info={}
    if has_alpha:result.info['aci_alpha']=alpha.resize(size,Image.Resampling.LANCZOS)
    if scale<1:result.info['aci_detail']=original
    result.info['aci_input_size']=[w,h]
    return result


def otsu(gray: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Otsu: maximizează varianța interclasă. Constanta are mască goală."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    count = np.cumsum(hist); sums = np.cumsum(hist * np.arange(256))
    denominator = count * (count[-1] - count)
    score = np.zeros(256); valid = denominator > 0
    score[valid] = (sums[-1] * count[valid] - sums[valid] * count[-1]) ** 2 / denominator[valid]
    return int(np.argmax(score)), hist.astype(int), score


def background_correct(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape; n = min(6, h, w)
    ys = np.linspace(0, h, n + 1).astype(int); xs = np.linspace(0, w, n + 1).astype(int)
    nodes = np.array([[np.percentile(gray[ys[i]:ys[i+1], xs[j]:xs[j+1]], 90)
                       for j in range(n)] for i in range(n)], dtype=float)
    xc = (xs[:-1] + xs[1:] - 1) / 2; yc = (ys[:-1] + ys[1:] - 1) / 2
    horizontal = np.array([np.interp(np.arange(w), xc, v) for v in nodes])
    surface = np.array([np.interp(np.arange(h), yc, horizontal[:, j]) for j in range(w)]).T
    corrected = np.clip(gray.astype(float) * 255 / np.maximum(surface, 1), 0, 255).astype('uint8')
    return corrected, surface.astype('uint8')


def stroke_width(mask: np.ndarray) -> float:
    runs_all = []; limit = max(3, min(mask.shape) // 12)
    for a in (mask, mask.T):
        for row in a:
            edges = np.flatnonzero(np.diff(np.pad(row.astype('int8'), 1)))
            runs = edges[1::2] - edges[::2]
            runs_all.extend(runs[(runs >= 1) & (runs <= limit)].tolist())
    return float(np.median(runs_all)) if runs_all else 1.


def close(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel == 1:
        return mask.copy()
    im = Image.fromarray(mask.astype('uint8') * 255)
    return np.asarray(im.filter(ImageFilter.MaxFilter(kernel)).filter(ImageFilter.MinFilter(kernel))) > 0


def fill_holes(mask: np.ndarray) -> np.ndarray:
    a = np.pad(mask, 1); outside = np.zeros(a.shape, dtype=bool)
    q = deque([(0, 0)]); outside[0, 0] = True; h, w = a.shape
    while q:
        y, x = q.popleft()
        for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
            if 0 <= ny < h and 0 <= nx < w and not a[ny,nx] and not outside[ny,nx]:
                outside[ny,nx] = True; q.append((ny,nx))
    return (~outside)[1:-1,1:-1]


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    remaining = mask.copy(); best = []; n = 0; h,w = mask.shape
    for y, x in zip(*np.where(mask)):
        if not remaining[y,x]: continue
        n += 1; q = deque([(int(y), int(x))]); remaining[y,x] = False; points = []
        while q:
            yy,xx = q.popleft(); points.append((yy,xx))
            for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                ny,nx=yy+dy,xx+dx
                if 0<=ny<h and 0<=nx<w and remaining[ny,nx]:
                    remaining[ny,nx]=False; q.append((ny,nx))
        if len(points)>len(best): best=points
    result=np.zeros(mask.shape,dtype=bool)
    if best:
        yy,xx=zip(*best); result[yy,xx]=True
    return result,n


def binary(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype('uint8') * 255)


def bbox(mask: np.ndarray, margin: int = 0) -> list[int]:
    yy,xx=np.where(mask)
    if len(xx)<8:
        raise ValueError('Nu am găsit suficiente urme de desen. Încearcă o fotografie mai clară sau alt exemplu.')
    h,w=mask.shape
    return [max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),
            min(w,int(xx.max())+1+margin),min(h,int(yy.max())+1+margin)]


def analyse(image: Image.Image,alpha=None) -> dict:
    gray=np.min(np.asarray(image),axis=2)
    corrected,background=background_correct(gray)
    threshold,hist,score=otsu(corrected)
    ink=corrected<=threshold
    if int(corrected.max())-int(corrected.min())<2: ink[:]=False
    stroke=stroke_width(ink)
    kernel=int(np.clip(round(stroke),1,7))
    if kernel%2==0: kernel+=1
    processed=close(ink,kernel)
    filled=fill_holes(processed)
    mask,n=components(filled)
    mask,segmentation=choose_mask(image,mask,alpha)
    # Localizarea alege o singură componentă; masca alpha a decupajului păstrează
    # ulterior toate piesele transparente, inclusiv cele deconectate.
    mask,_=components(mask)
    return dict(gray=gray,corrected=corrected,background=background,threshold=threshold,
                hist=hist,score=score,ink=ink,stroke=stroke,kernel=kernel,
                processed=processed,filled=filled,mask=mask,components=n,segmentation=segmentation)


def pose_proposal(mask: np.ndarray) -> list[list[float]]:
    """Propunere antropometrică pe bbox; NU este inferență neuronală."""
    x0,y0,x1,y1=bbox(mask); w=x1-x0; h=y1-y0
    normalized=[(.5,.64),(.5,.35),(.5,.14),(.34,.34),(.2,.44),(.08,.40),
                (.66,.34),(.8,.44),(.92,.40),(.4,.65),(.36,.80),(.31,.96),
                (.6,.65),(.64,.80),(.69,.96)]
    return [[round(x0+x*w,2),round(y0+y*h,2)] for x,y in normalized]


def reference_pose(source_size: tuple[int,int], crop: list[int],output_size=None) -> list[list[float]]:
    # Coordonate adnotate manual pe garlic_roi; reutilizate doar pe variațiile sale.
    normalized=[(.55,.69),(.54,.40),(.54,.19),(.33,.34),(.25,.25),(.17,.16),
                (.77,.32),(.81,.22),(.85,.12),(.41,.68),(.39,.79),(.37,.87),
                (.69,.68),(.72,.79),(.76,.87)]
    w,h=source_size; x0,y0,_,_=crop
    sx,sy=(1,1) if output_size is None else (output_size[0]/(crop[2]-x0),output_size[1]/(crop[3]-y0))
    return [[round((x*w-x0)*sx,2),round((y*h-y0)*sy,2)] for x,y in normalized]


def triangulate(points: np.ndarray) -> list[list[int]]:
    """Bowyer-Watson, circumscripții calculate vectorizat; fără SciPy."""
    n=len(points); lo=points.min(axis=0); hi=points.max(axis=0)
    span=max(float((hi-lo).max()),1); center=(lo+hi)/2
    p=np.vstack([points,center+[-20*span,-10*span],center+[20*span,-10*span],center+[0,20*span]])
    triangles=np.array([[n,n+1,n+2]],dtype=int)
    for i in range(n):
        a,b,c=p[triangles[:,0]],p[triangles[:,1]],p[triangles[:,2]]
        d=2*(a[:,0]*(b[:,1]-c[:,1])+b[:,0]*(c[:,1]-a[:,1])+c[:,0]*(a[:,1]-b[:,1]))
        valid=np.abs(d)>1e-9
        aa=(a*a).sum(1);bb=(b*b).sum(1);cc=(c*c).sum(1)
        divisor=np.where(valid,d,1)
        ux=(aa*(b[:,1]-c[:,1])+bb*(c[:,1]-a[:,1])+cc*(a[:,1]-b[:,1]))/divisor
        uy=(aa*(c[:,0]-b[:,0])+bb*(a[:,0]-c[:,0])+cc*(b[:,0]-a[:,0]))/divisor
        radius=(ux-a[:,0])**2+(uy-a[:,1])**2
        dist=(ux-p[i,0])**2+(uy-p[i,1])**2
        bad=valid&(dist<=radius+1e-7)
        edges={}
        for t in triangles[bad]:
            for j,k in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
                key=tuple(sorted((int(j),int(k))));edges[key]=edges.get(key,0)+1
        additions=[[a,b,i] for (a,b),count in edges.items() if count==1]
        triangles=np.concatenate([triangles[~bad],np.array(additions,dtype=int).reshape(-1,3)],axis=0)
    return triangles[(triangles<n).all(axis=1)].tolist()


def make_mesh(mask: np.ndarray, joints: list[list[float]]) -> dict:
    h,w=mask.shape
    if mask.sum()<12: raise ValueError('Masca este prea mică pentru o suprafață. Corectează pasul 5.')
    # Mesh-ul acoperă și marginea măștii; transparența texturii păstrează conturul.
    expanded=np.asarray(binary(mask).filter(ImageFilter.MaxFilter(5)))>0
    eroded=np.asarray(binary(expanded).filter(ImageFilter.MinFilter(3)))>0
    boundary=expanded&~eroded
    yy,xx=np.where(boundary); points=[]; used=set()
    spacing=max(4,round(max(h,w)/90))
    for x,y in zip(xx,yy):
        key=(int(x)//spacing,int(y)//spacing)
        if key not in used: points.append([float(x),float(y)]);used.add(key)
    grid=max(12,round(max(h,w)/22))
    for y in range(grid//2,h,grid):
        for x in range(grid//2,w,grid):
            if expanded[y,x]: points.append([float(x),float(y)])
    for x,y in joints:
        if 0<=x<w and 0<=y<h and expanded[int(y),int(x)]: points.append([float(x),float(y)])
    p=np.unique(np.round(np.array(points),2),axis=0)
    if len(p)>1500: raise ValueError('Contur prea complex. Simplifică sau corectează masca înainte de triangulare.')
    candidates=triangulate(p); kept=[]
    for tri in candidates:
        a,b,c=p[tri]
        cross=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if abs(cross)<.01: continue
        probes=np.array([(a+b+c)/3,(a+b)/2,(b+c)/2,(c+a)/2])
        xy=np.rint(probes).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
        if expanded[xy[:,1],xy[:,0]].all():kept.append(tri)
    if not kept: raise ValueError('Nu am putut construi triunghiuri valide din această mască.')
    used_indices=sorted(set(x for t in kept for x in t)); remap={j:i for i,j in enumerate(used_indices)}
    p=p[used_indices];kept=[[remap[j] for j in t] for t in kept]
    j=np.asarray(joints,dtype=float); distances=[]
    for a,b in BONES:
        v=j[b]-j[a];den=max(float(v@v),1e-6)
        t=np.clip(((p-j[a])@v)/den,0,1)
        delta=p-(j[a]+t[:,None]*v)
        distances.append(np.sqrt((delta*delta).sum(1)))
    d=np.stack(distances,axis=1)
    # Două oase apropiate; ponderi nenegative normalizate.
    nearest=np.argsort(d,axis=1)[:,:2]
    weights=[]
    for i,bs in enumerate(nearest):
        raw=1/np.maximum(d[i,bs],2.)**4;raw/=raw.sum()
        weights.append([[int(b),round(float(v),8)] for b,v in zip(bs,raw)])
    return dict(vertices=p.tolist(),triangles=kept,weights=weights,bones=BONES,
                joints=joints,parents=PARENTS,joint_names=JOINT_NAMES,
                method='Delaunay filtrată pe mască + linear blend skinning',
                point_spacing=spacing,grid_spacing=grid)


@dataclass
class DrawingSession:
    source: Image.Image
    name: str='Desen încărcat'
    example: str|None=None
    crop: list[int]|None=None
    images: dict=field(default_factory=dict)
    meta: dict=field(default_factory=dict)
    computed: int=0
    analysis: dict|None=None
    joints: list|None=None
    mesh: dict|None=None
    pose_kind: str=''
    pose_method: str='auto'
    anatomy: dict|None=None
    body_override: np.ndarray|None=None
    labels_override: np.ndarray|None=None
    joint_quality: list=field(default_factory=list)
    source_alpha: Image.Image|None=None
    source_detail: Image.Image|None=None
    crop_alpha: Image.Image|None=None
    restored_crop: Image.Image|None=None
    restored_crop_alpha: Image.Image|None=None
    restored_crop_bounds: list|None=None
    pose_backend: str=''
    pose_evidence: dict|None=None
    settings: dict=field(default_factory=lambda:dict(render_mode='puppet',motion='cmu_07_01',motion_projection='auto',show_motion_source=True,comparison=False,limbs_front=False,smooth_joints=True))
    revision: int=0
    created: float=field(default_factory=time.monotonic)

    def __post_init__(self):
        self.source_alpha=self.source.info.pop('aci_alpha',None)
        self.source_detail=self.source.info.pop('aci_detail',None)
        input_size=self.source.info.pop('aci_input_size',list(self.source.size))
        self.images['original']=self.source
        self.meta={'normalization':'lățime max. 400 px; înălțime max. 700 px',
                   'source_size':list(self.source.size),'input_size':input_size,'input_has_alpha':self.source_alpha is not None,'fixed_rules':{
                       'background_grid':6,'background_percentile':90,
                       'kernel_bounds':[1,7],'connectivity':8,'fill_connectivity':4},
                   'stage_seconds':{},'mask_edited':False}

    def invalidate(self, first: int):
        self.computed=min(self.computed,first-1);self.revision+=1
        image_stages={'crop':1,'gray':2,'corrected':2,'background':2,'ink':3,
                      'closed':4,'filled':5,'mask':5,'texture':5,'overlay':5,
                      'distance':6,'body_mask':6,'body_overlay':6,'parts':6,'skeleton_map':6,'part_labels':6,
                      'neural_input':6,**{f'heatmap_{i}':6 for i in range(15)},
                      'limbs_texture':7,**{f'piece_{i}':7 for i in range(9)}}
        for key in list(self.images):
            if image_stages.get(key,0)>=first:self.images.pop(key)
        if first<=1:self.crop_alpha=None
        if first<=2:self.analysis=None
        meta_stages={'bbox':1,'gray_method':2,'threshold':3,'histogram':3,
                     'stroke_width_px':3,'kernel':3,'morphology':4,
                     'foreground_fraction':5,'components_before_selection':5,'segmentation':5,
                     'warnings':5,'pose_diagnostics':6,'body_pixels':6,'part_pixels':6,'neural_pose':6,'pose_comparison':6,
                     'vertices':7,'triangles':7,'animation':8,'report':9}
        for key in list(self.meta):
            if meta_stages.get(key,0)>=first:self.meta.pop(key)
        if first<=5:self.meta['mask_edited']=False
        timings=self.meta.get('stage_seconds',{})
        for key in list(timings):
            if int(key)>=first:timings.pop(key)
        if first<=5:self.body_override=None;self.labels_override=None
        if first<=6:
            self.joints=None;self.pose_kind='';self.anatomy=None;self.joint_quality=[]
            self.pose_backend='';self.pose_evidence=None
        if first<=7:self.mesh=None

    def run(self, target: int):
        if not 0<=target<=9: raise ValueError('Etapă invalidă.')
        for stage in range(self.computed+1,target+1):
            started=time.perf_counter()
            if stage==1:
                a=analyse(self.source,self.source_alpha)
                if self.crop is None:
                    self.crop=bbox(a['mask'],margin=max(7,round(max(self.source.size)*.025)))
                    self.meta['localization']='Estimare geometrică prin '+a['segmentation']['description']+'; fără detector neuronal'
                if self.restored_crop is not None and self.crop==self.restored_crop_bounds:
                    self.images['crop']=self.restored_crop.copy();self.crop_alpha=self.restored_crop_alpha
                elif self.source_detail is not None:
                    # Dreptunghiul este în coordonatele imaginii de ansamblu.
                    # Revenim în original înainte de a redimensiona personajul.
                    sx=self.source_detail.width/self.source.width;sy=self.source_detail.height/self.source.height
                    x0,y0,x1,y1=self.crop
                    raw=self.source_detail.crop((int(np.floor(x0*sx)),int(np.floor(y0*sy)),int(np.ceil(x1*sx)),int(np.ceil(y1*sy))))
                    scale=min(1,400/raw.width,700/raw.height)
                    size=(max(8,round(raw.width*scale)),max(8,round(raw.height*scale)))
                    white=Image.new('RGBA',raw.size,'white')
                    self.images['crop']=Image.alpha_composite(white,raw).convert('RGB').resize(size,Image.Resampling.LANCZOS)
                    self.crop_alpha=raw.getchannel('A').resize(size,Image.Resampling.LANCZOS) if self.source_alpha is not None else None
                else:
                    self.images['crop']=self.source.crop(tuple(self.crop))
                    self.crop_alpha=self.source_alpha.crop(tuple(self.crop)) if self.source_alpha is not None else None
                self.meta['bbox']=self.crop
                self.meta['crop_size']=list(self.images['crop'].size)
            elif stage==2:
                gray=np.min(np.asarray(self.images['crop']),axis=2)
                corrected,background=background_correct(gray)
                self.analysis=dict(gray=gray,corrected=corrected,background=background);a=self.analysis
                for k in ('gray','corrected','background'):self.images[k]=Image.fromarray(a[k])
                self.meta['gray_method']='min(R,G,B), pe canale de 8 biți'
            elif stage==3:
                a=self.analysis
                threshold,hist,score=otsu(a['corrected']);ink=a['corrected']<=threshold
                if int(a['corrected'].max())-int(a['corrected'].min())<2:ink[:]=False
                stroke=stroke_width(ink);kernel=int(np.clip(round(stroke),1,7))
                if kernel%2==0:kernel+=1
                a.update(threshold=threshold,hist=hist,score=score,ink=ink,stroke=stroke,kernel=kernel)
                self.images['ink']=binary(a['ink'])
                self.meta.update(threshold=a['threshold'],histogram=a['hist'].tolist(),
                                 stroke_width_px=round(a['stroke'],2),kernel=a['kernel'])
            elif stage==4:
                self.analysis['processed']=close(self.analysis['ink'],self.analysis['kernel'])
                self.images['closed']=binary(self.analysis['processed'])
                self.meta['morphology']='închidere = dilatare urmată de eroziune; un nucleu pătrat estimat'
            elif stage==5:
                a=self.analysis
                a['filled']=fill_holes(a['processed']);legacy,a['components']=components(a['filled'])
                a['mask'],segmentation=choose_mask(self.images['crop'],legacy,self.crop_alpha)
                self.images['filled']=binary(a['filled']);self.set_mask(a['mask'],False)
                self.meta['components_before_selection']=a['components']
                self.meta['segmentation']=segmentation
            elif stage==6:
                self.estimate_pose(self.pose_method)
            elif stage==7:
                self.mesh=make_mesh(np.asarray(self.images['mask'])>0,self.joints)
                bind_protected(self.mesh,self.anatomy)
                pieces,spec=puppet_images(self.images['texture'],self.anatomy,self.joints)
                self.images.update(pieces);self.mesh['puppet_parts']=spec
                self.mesh['joint_bridges']=puppet_bridges(self.images['texture'],self.anatomy,self.joints)
                self.meta['vertices']=len(self.mesh['vertices']);self.meta['triangles']=len(self.mesh['triangles'])
            elif stage==8:
                self.meta['animation']='CMU motion capture 3D sau proceduri 2D pentru comparație; marionetă rigidă / corp protejat / LBS'
            elif stage==9:
                self.meta['report']='Disponibile toate imaginile intermediare, rig-ul, parametrii și raportul.'
            self.meta['stage_seconds'][str(stage)]=round(time.perf_counter()-started,4)
            self.computed=stage

    def set_mask(self, mask: np.ndarray, edited=True):
        w,h=self.images['crop'].size
        if mask.shape!=(h,w):raise ValueError('Masca trebuie să aibă exact dimensiunea decupajului.')
        self.images['mask']=binary(mask)
        texture=self.images['crop'].convert('RGBA');texture.putalpha(binary(mask));self.images['texture']=texture
        overlay=np.asarray(self.images['crop']).astype(float)
        overlay[mask]=overlay[mask]*.58+np.array([20,158,130])*.42
        self.images['overlay']=Image.fromarray(overlay.astype('uint8'))
        self.meta['foreground_fraction']=round(float(mask.mean()),5)
        self.meta['mask_edited']=edited
        if edited:self.meta['segmentation']={'method':'manual','description':'Mască corectată manual în Selecție','note':'Corecțiile au prioritate față de masca automată.'}
        warnings=[]
        if mask.mean()<.002:warnings.append('Masca este goală sau foarte mică. Corectează selecția înainte de rig.')
        if mask.mean()>.75:warnings.append('Masca ocupă peste 75% din decupaj. Verifică dacă a fost inclus fundalul.')
        if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
            warnings.append('Masca atinge marginea: verifică dacă personajul este tăiat.')
        self.meta['warnings']=warnings
        self.body_override=None;self.labels_override=None

    def estimate_pose(self,method='auto'):
        mask=np.asarray(self.images['mask'])>0;geometry=None;result=None;fallback=None
        if method not in ('auto','neural','official','finetuned','shape','reference','proportions'):raise ValueError('Metodă necunoscută.')
        selected=method
        if selected=='auto':selected='neural' if model_status()['available'] else 'shape'
        if selected in ('neural','official','finetuned'):
            try:
                result=predict_pose(self.images['crop']) if selected=='neural' else predict_pose(self.images['crop'],choice=selected)
                selected='neural'
            except (ValueError,RuntimeError,OSError,ImportError) as error:
                if method!='auto':raise ValueError('Predictorul neuronal nu a putut rula: '+str(error)) from error
                selected='shape';fallback='Modelul nu a putut rula; folosim geometria. '+str(error)
        # Calculăm înainte de a înlocui rezultatul existent, pentru erori recuperabile.
        if selected=='neural':
            joints=result['joints'];kind=result.get('model_name','Model oficial AnimatedDrawings · ResNet-50')+' · 17 heatmap-uri → 15 controale'
            quality=result['quality']
        elif selected=='shape':
            joints,geometry=shape_pose(mask)
            kind='Repere estimate din siluetă: distanță, schelet și trasee; verifică punctele ambigue'
            quality=geometry['quality']
        elif selected=='reference' and self.example and self.example.startswith('garlic_'):
            joints=reference_pose(self.source.size,self.crop,self.images['crop'].size)
            kind='Referință didactică: repere adnotate manual pe Garlic';quality=[1.]*15
        elif selected=='proportions':
            joints=pose_proposal(mask);kind='Referință veche: proporții în dreptunghi, fără analiza siluetei';quality=[.25]*15
        else:raise ValueError('Metodă de estimare indisponibilă pentru acest desen.')
        self.invalidate(7)
        for key in list(self.images):
            if key.startswith('heatmap_') or key=='neural_input':self.images.pop(key)
        self.meta.pop('neural_pose',None);self.meta.pop('pose_comparison',None)
        self.joints=joints;self.pose_kind=kind;self.joint_quality=quality;self.pose_backend=selected
        self.pose_evidence=None
        if result:
            self.pose_evidence={k:result[k] for k in ['keypoints','names','control_order','notes','seconds','source','quality_meaning']}
            self.pose_evidence.update({k:result[k] for k in ['model_id','model_name','model_profile'] if k in result})
            self.pose_evidence['predicted_controls']=joints
            self.meta['neural_pose']=self.pose_evidence
            self.images['neural_input']=result['input_image']
            for control in range(15):self.images[f'heatmap_{control}']=heatmap_overlay(self.images['crop'],result,control)
            baseline,_=shape_pose(mask)
            delta=np.linalg.norm(np.array(joints)-baseline,axis=1)
            self.meta['pose_comparison']=dict(geometric_joints=baseline,mean_difference_px=round(float(delta.mean()),2),
                meaning='Dezacord între două metode; nu eroare față de adevărul de referință')
        w,h=self.images['crop'].size
        self.joints=[[min(w-1,max(0,x)),min(h-1,max(0,y))] for x,y in self.joints]
        self.pose_method=method;self.update_structure(geometry)
        if fallback:self.meta['pose_diagnostics']['notes'].insert(0,fallback)
        if method=='auto' and selected=='shape' and not model_status()['available']:
            self.meta['pose_diagnostics']['notes'].insert(0,'Model neuronal indisponibil: rezultat geometric. '+model_status()['message'])

    def update_structure(self,geometry=None):
        mask=np.asarray(self.images['mask'])>0
        anatomical=self.pose_backend=='neural'
        if anatomical:
            anatomical=bool(np.linalg.norm(np.array(self.joints[1])-self.joints[0])>=.06*mask.shape[0])
            if self.pose_evidence and self.joint_quality:
                keypoints=np.asarray(self.pose_evidence['keypoints'])
                anatomical=anatomical and bool(np.min(keypoints[[5,6,11,12],2])>=.35)
        self.anatomy=structure(mask,self.joints,self.body_override,self.labels_override,geometry,anatomical=anatomical)
        self.images.update(diagnostic_images(self.images['texture'],self.anatomy))
        checks=self.anatomy['checks'];notes=list(geometry['notes']) if geometry is not None else []
        if self.pose_evidence:notes+=self.pose_evidence['notes']
        if self.pose_backend=='neural' and not anatomical:notes.append('Trunchiul neuronal este nesigur; zona rigidă folosește propunerea geometrică. Corectează corpul dacă este necesar.')
        for check in checks:
            if not check['on_mask'] and check['distance_to_mask']>2:notes.append(f'{JOINT_NAMES[check["joint"]]}: la {check["distance_to_mask"]} px de siluetă; verifică poziția.')
        self.meta['pose_diagnostics']=dict(notes=notes,checks=checks,quality=self.joint_quality,
            quality_meaning=self.pose_evidence['quality_meaning'] if self.pose_evidence else 'Scor euristic de consistență geometrică; nu probabilitate neuronală',
            backend=self.pose_backend,body_method='Umeri + șolduri, grosime și partiționare geodezică' if anatomical else 'Grosime și partiționare geodezică',
            body_manual=self.body_override is not None,parts_manual=self.labels_override is not None,
            endpoints=geometry.get('endpoints') if geometry else None,neck_y=geometry.get('neck_y') if geometry else None)
        self.meta['body_pixels']=int(self.anatomy['body'].sum())
        self.meta['part_pixels']=[int((self.anatomy['labels']==i).sum()) for i in range(9)]
        comparison=self.meta.get('pose_comparison')
        if comparison:
            delta=np.linalg.norm(np.array(self.joints)-comparison['geometric_joints'],axis=1)
            comparison['mean_difference_px']=round(float(delta.mean()),2)

    def state(self, sid: str) -> dict:
        return dict(id=sid,name=self.name,example=self.example,computed=self.computed,revision=self.revision,
                    meta=self.meta,joints=self.joints,pose_kind=self.pose_kind,mesh=self.mesh,
                    pose_method=self.pose_method,pose_backend=self.pose_backend,model=model_status(),settings=self.settings,parts=PARTS,
                    joint_names=JOINT_NAMES,parents=PARENTS,bones=BONES,
                    images={k:f'/api/image/{sid}/{k}?r={self.revision}' for k in self.images})

    def project(self) -> dict:
        return dict(format='aci-drawing-lab',version=VERSION,name=self.name,example=self.example,
                    source=data_url(self.source),crop=self.crop,
                    source_alpha=data_url(self.source_alpha) if self.source_alpha is not None else None,
                    crop_image=data_url(self.images['crop']) if 'crop' in self.images else None,
                    crop_alpha=data_url(self.crop_alpha) if self.crop_alpha is not None else None,
                    mask=data_url(self.images['mask']) if 'mask' in self.images else None,
                    joints=self.joints,pose_kind=self.pose_kind,pose_method=self.pose_method,pose_backend=self.pose_backend,
                    pose_evidence=self.pose_evidence,joint_quality=self.joint_quality,
                    body_override=data_url(binary(self.body_override)) if self.body_override is not None else None,
                    parts_override=data_url(Image.fromarray(np.where(self.labels_override>=0,self.labels_override,255).astype('uint8'))) if self.labels_override is not None else None,
                    settings=self.settings,computed=self.computed,meta=self.meta)

    def bundle(self) -> bytes:
        out=BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for k,im in self.images.items():z.writestr(f'imagini/{k}.png',png_bytes(im))
            z.writestr('proiect.json',json.dumps(self.project(),ensure_ascii=False,indent=2))
            z.writestr('parametri.json',json.dumps(self.meta,ensure_ascii=False,indent=2))
            z.writestr('rig.json',json.dumps(self.mesh or {'joints':self.joints},ensure_ascii=False,indent=2))
            motion=self.settings.get('motion','')
            source=next((c for c in motion_catalog() if c['id']==motion),None)
            if source:
                z.writestr('miscare/'+motion+'.json',json.dumps(motion_clip(motion),ensure_ascii=False,separators=(',',':')))
                z.writestr('miscare/SURSA.json',json.dumps(source,ensure_ascii=False,indent=2))
            z.writestr('CITESTE.txt','Export din ACI Drawing Lab. Deschide proiect.json prin Importă proiect.\n'
                       'Localizare geometrică, Otsu, morfologie, repere editabile, Delaunay, LBS.\n'
                       'Predictor neuronal oficial opțional; marionetă rigidă locală; CMU mocap sau proceduri 2D. Fără ARAP.\n')
        return out.getvalue()


def restore_project(p: dict) -> DrawingSession:
    if p.get('format')!='aci-drawing-lab':raise ValueError('Format de proiect necunoscut.')
    s=DrawingSession(normalize(decode_image(p['source'])),str(p.get('name','Proiect importat'))[:100],p.get('example'))
    if p.get('crop') is not None:set_crop(s,p['crop'])
    if p.get('source_alpha'):
        alpha=decode_image(p['source_alpha']).convert('L')
        if alpha.size!=s.source.size:raise ValueError('Alpha nu corespunde dimensiunii imaginii sursă.')
        s.source_alpha=alpha;s.meta['input_has_alpha']=True
    if p.get('crop_image'):
        detail=decode_image(p['crop_image']).convert('RGB')
        if detail.width>400 or detail.height>700:raise ValueError('Decupajul salvat trebuie să aibă maximum 400 × 700 px.')
        s.restored_crop=detail;s.restored_crop_bounds=s.crop.copy() if s.crop else None
        if p.get('crop_alpha'):
            alpha=decode_image(p['crop_alpha']).convert('L')
            if alpha.size!=detail.size:raise ValueError('Alpha decupajului are dimensiuni greșite.')
            s.restored_crop_alpha=alpha
    target=int(p.get('computed',0))
    s.run(min(5,target))
    if p.get('mask') and target>=5:
        s.set_mask(np.asarray(decode_image(p['mask']).convert('L'))>127,bool(p.get('meta',{}).get('mask_edited',False)))
    if target>=6:
        # Un proiect salvat se deschide și fără model. Nu rerulăm inferența ca să
        # suprascriem reperele deja corectate de utilizator.
        if p.get('joints') is not None:
            s.pose_method=p.get('pose_method','shape');s.pose_backend=p.get('pose_backend','shape')
            s.pose_evidence=copy.deepcopy(p.get('pose_evidence'));s.joint_quality=list(p.get('joint_quality',[]))
            if s.pose_evidence:s.meta['neural_pose']=s.pose_evidence
            comparison=p.get('meta',{}).get('pose_comparison')
            if isinstance(comparison,dict):
                proposed=np.asarray(comparison.get('geometric_joints',[]),dtype=float)
                if proposed.shape==(15,2) and np.isfinite(proposed).all():s.meta['pose_comparison']=copy.deepcopy(comparison)
            s.computed=6;set_pose(s,p['joints'],edited=False);s.pose_kind=str(p.get('pose_kind','Repere importate'))[:200]
        else:
            s.pose_method=p.get('pose_method','auto');s.run(6)
        if p.get('body_override'):
            set_body(s,np.asarray(decode_image(p['body_override']).convert('L'))>127)
        if p.get('parts_override'):
            set_parts(s,np.asarray(decode_image(p['parts_override']).convert('L')))
    set_settings(s,p.get('settings',{}))
    s.run(target)
    return s


def set_crop(s: DrawingSession, values):
    if not isinstance(values,list) or len(values)!=4:raise ValueError('Dreptunghi invalid.')
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Coordonate invalide.')
    w,h=s.source.size;x0,y0,x1,y1=np.rint(a).astype(int).tolist()
    x0=max(0,min(w,x0));x1=max(0,min(w,x1));y0=max(0,min(h,y0));y1=max(0,min(h,y1))
    if x1-x0<8 or y1-y0<8:raise ValueError('Decupajul trebuie să aibă cel puțin 8×8 pixeli.')
    s.crop=[x0,y0,x1,y1];s.meta['localization']='Dreptunghi corectat de utilizator'
    s.invalidate(1)


def set_pose(s: DrawingSession, joints,edited=True):
    a=np.asarray(joints,dtype=float)
    if a.shape!=(15,2) or not np.isfinite(a).all():raise ValueError('Sunt necesare 15 repere cu coordonate finite.')
    w,h=s.images['crop'].size
    if (a[:,0]<0).any() or (a[:,0]>=w).any() or (a[:,1]<0).any() or (a[:,1]>=h).any():
        raise ValueError('Reperele trebuie să rămână în imagine.')
    s.joints=a.round(3).tolist();s.pose_kind='Repere corectate de utilizator';s.invalidate(7)
    if edited:s.joint_quality=[]
    s.meta.pop('pose_diagnostics',None);s.update_structure()


def set_body(s:DrawingSession,mask):
    s.run(6)
    if mask is None:s.body_override=None
    else:
        if mask.shape!=np.asarray(s.images['mask']).shape:raise ValueError('Corpul trebuie să aibă dimensiunile măștii.')
        body=mask&(np.asarray(s.images['mask'])>0)
        if body.sum()<8:raise ValueError('Selectează cel puțin o mică regiune a corpului central.')
        s.body_override=body
    s.labels_override=None;s.invalidate(7);s.update_structure()


def set_parts(s:DrawingSession,labels):
    s.run(6);mask=np.asarray(s.images['mask'])>0
    if labels is None:s.labels_override=None;s.body_override=None
    else:
        if labels.shape!=mask.shape:raise ValueError('Etichetele trebuie să aibă dimensiunea măștii.')
        if np.any((labels[mask]<0)|(labels[mask]>8)):raise ValueError('Fiecare pixel al personajului trebuie să aparțină unei piese 0…8.')
        # PNG-ul folosește uint8 (255 = fundal), dar etichetele interne folosesc
        # -1 pentru fundal. Convertim înainte de a atribui valoarea negativă.
        proposed=labels.astype('int16',copy=True);proposed[~mask]=-1;body=proposed==0
        if body.sum()<8:raise ValueError('Păstrează o regiune pentru corpul central.')
        s.labels_override=proposed;s.body_override=body
    s.invalidate(7);s.update_structure()


def set_settings(s:DrawingSession,settings):
    if not isinstance(settings,dict):raise ValueError('Setări invalide.')
    mode=settings.get('render_mode',s.settings['render_mode']);motion=settings.get('motion',s.settings['motion'])
    projection=settings.get('motion_projection',s.settings.get('motion_projection','auto'))
    if mode not in ('baseline','protected','puppet') or motion not in ('wave','walk','dance',*(c['id'] for c in motion_catalog())):
        raise ValueError('Mod de animație sau mișcare necunoscută.')
    if projection not in ('auto','front','side'):raise ValueError('Proiecție de mișcare necunoscută.')
    s.settings=dict(render_mode=mode,motion=motion,motion_projection=projection,
        show_motion_source=bool(settings.get('show_motion_source',s.settings.get('show_motion_source',True))),
        comparison=bool(settings.get('comparison',False)),limbs_front=bool(settings.get('limbs_front',False)),
        smooth_joints=bool(settings.get('smooth_joints',s.settings.get('smooth_joints',True))))
