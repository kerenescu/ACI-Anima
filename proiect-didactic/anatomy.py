"""Geometrie inspectabilă pentru corp, articulații și piese de marionetă.

Fără rețea neuronală. Distanță chamfer 8 vecini, subțiere Zhang–Suen,
trasee geodezice și partiționare pe mască. Scorurile sunt euristice.
"""
from __future__ import annotations
from collections import deque
import heapq
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PARTS = [
    {'id':0,'name':'Corp central + cap','bone':0,'chain':[],'color':[30,135,112]},
    {'id':1,'name':'Braț stânga ecran','bone':3,'chain':[3,4],'color':[224,126,68]},
    {'id':2,'name':'Antebraț stânga ecran','bone':4,'chain':[3,4],'color':[241,184,86]},
    {'id':3,'name':'Braț dreapta ecran','bone':6,'chain':[6,7],'color':[76,131,202]},
    {'id':4,'name':'Antebraț dreapta ecran','bone':7,'chain':[6,7],'color':[113,185,224]},
    {'id':5,'name':'Coapsă stânga ecran','bone':9,'chain':[9,10],'color':[158,101,181]},
    {'id':6,'name':'Gambă stânga ecran','bone':10,'chain':[9,10],'color':[202,144,200]},
    {'id':7,'name':'Coapsă dreapta ecran','bone':12,'chain':[12,13],'color':[153,157,64]},
    {'id':8,'name':'Gambă dreapta ecran','bone':13,'chain':[12,13],'color':[201,199,105]},
]
CHAINS=[(3,4,5),(6,7,8),(9,10,11),(12,13,14)]
NEIGHBORS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Distanță chamfer la exterior, incluzând chenarul imaginii. Unități px."""
    a=np.pad(np.where(mask,1.e6,0.).astype('float32'),1)
    idx=np.arange(a.shape[1],dtype='float32');diag=np.float32(math.sqrt(2))
    for y in range(1,a.shape[0]-1):
        row=np.minimum(a[y],a[y-1]+1)
        row[1:]=np.minimum(row[1:],a[y-1,:-1]+diag)
        row[:-1]=np.minimum(row[:-1],a[y-1,1:]+diag)
        a[y]=np.minimum.accumulate(row-idx)+idx
    for y in range(a.shape[0]-2,0,-1):
        row=np.minimum(a[y],a[y+1]+1)
        row[1:]=np.minimum(row[1:],a[y+1,:-1]+diag)
        row[:-1]=np.minimum(row[:-1],a[y+1,1:]+diag)
        a[y]=(np.minimum.accumulate(row[::-1]-idx)+idx)[::-1]
    return np.maximum(a[1:-1,1:-1],0)*mask


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen, până la convergență. Nu este recunoaștere anatomică."""
    a=np.pad(mask.astype('uint8'),1)
    for _ in range(max(mask.shape)):
        changed=False
        for phase in (0,1):
            p=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],
               a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            count=sum(p);transitions=sum((p[i]==0)&(p[(i+1)%8]==1) for i in range(8))
            if phase==0:keep=(p[0]*p[2]*p[4]==0)&(p[2]*p[4]*p[6]==0)
            else:keep=(p[0]*p[2]*p[6]==0)&(p[0]*p[4]*p[6]==0)
            remove=(a[1:-1,1:-1]==1)&(count>=2)&(count<=6)&(transitions==1)&keep
            if remove.any():a[1:-1,1:-1][remove]=0;changed=True
        if not changed:break
    return a[1:-1,1:-1]>0


def nearest_point(points: np.ndarray, target) -> np.ndarray:
    return points[np.argmin(((points-np.asarray(target))**2).sum(axis=1))]


def skeleton_paths(skeleton: np.ndarray, center) -> tuple[list,np.ndarray]:
    yy,xx=np.where(skeleton);points=np.stack([xx,yy],axis=1)
    if not len(points):return [],np.array(center)
    root=nearest_point(points,center);rx,ry=map(int,root)
    h,w=skeleton.shape;dist=np.full((h,w),np.inf);dist[ry,rx]=0
    previous={};heap=[(0.,ry,rx)]
    while heap:
        d,y,x=heapq.heappop(heap)
        if d>dist[y,x]+1.e-8:continue
        for dy,dx in NEIGHBORS:
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and skeleton[ny,nx]:
                nd=d+(math.sqrt(2) if dy and dx else 1.)
                if nd<dist[ny,nx]-1.e-8:
                    dist[ny,nx]=nd;previous[(ny,nx)]=(y,x);heapq.heappush(heap,(nd,ny,nx))
    out=[]
    for x,y in points:
        if not np.isfinite(dist[y,x]):continue
        degree=sum(0<=y+dy<h and 0<=x+dx<w and skeleton[y+dy,x+dx] for dy,dx in NEIGHBORS)
        if degree<=1 and dist[y,x]>6:
            route=[];current=(int(y),int(x))
            while current!=(ry,rx):
                route.append([current[1],current[0]]);current=previous[current]
            route.append([rx,ry]);out.append({'tip':[int(x),int(y)],'length':float(dist[y,x]),'route':route})
    return out,root


def shape_pose(mask: np.ndarray) -> tuple[list,dict]:
    yy,xx=np.where(mask)
    if len(xx)<12:raise ValueError('Masca este prea mică pentru identificarea corpului.')
    dt=distance_transform(mask);skeleton=thin(mask)
    h,w=mask.shape;x0,x1=float(xx.min()),float(xx.max());y0,y1=float(yy.min()),float(yy.max())
    bw=max(x1-x0,1.);bh=max(y1-y0,1.)
    wide=dt>=max(2.,float(dt.max())*.28)
    wy,wx=np.where(wide)
    cx=float(np.median(wx)) if len(wx) else (x0+x1)/2
    cy=float(np.median(wy)) if len(wy) else (y0+y1)/2
    # Un cap disproporționat nu trebuie confundat automat cu toracele.
    # Căutăm o îngustare centrală sub zona lată superioară (posibil gât).
    widths=[]
    for y in range(h):
        xs=np.flatnonzero(mask[y]);run=0
        if len(xs):
            cuts=np.r_[0,np.flatnonzero(np.diff(xs)>1)+1,len(xs)]
            segments=[xs[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]
            best=min(segments,key=lambda v:abs(float(v.mean())-cx))
            if abs(float(best.mean())-cx)<bw*.18:run=len(best)
        widths.append(run)
    widths=np.array(widths,dtype=float);window=max(3,round(bh*.025))
    smooth=np.convolve(widths,np.ones(window)/window,mode='same')
    top_end=max(1,min(h,round(y0+bh*.53)));peak=int(np.argmax(smooth[:top_end]))
    neck=None
    if smooth[peak]>bw*.42:
        for y in range(peak+max(3,round(bh*.07)),min(h,round(y0+bh*.67))):
            if bw*.025<smooth[y]<smooth[peak]*.32:
                neck=y;break
    if neck is not None:
        cy=neck+(y1-neck)*.26
        row=mask[int(np.clip(round(cy),0,h-1))];rx=np.flatnonzero(row)
        if len(rx):cx=float(rx[np.argmin(abs(rx-cx))])
    endpoints,root=skeleton_paths(skeleton,[cx,cy])
    local_radius=float(dt[int(root[1]),int(root[0])])
    default=[(.5,.64),(.5,.35),(.5,.14),(.34,.34),(.2,.44),(.08,.40),
             (.66,.34),(.8,.44),(.92,.40),(.4,.65),(.36,.8),(.31,.96),
             (.6,.65),(.64,.8),(.69,.96)]
    foreground=np.stack([xx,yy],axis=1)
    joints=[nearest_point(foreground,[x0+x*bw,y0+y*bh]).astype(float).tolist() for x,y in default]
    if neck is not None:
        remain=y1-neck;shoulder=neck+remain*.14;hip=neck+remain*.50
        adaptive={0:[cx,hip],1:[cx,shoulder],2:[cx,max(y0,neck-bh*.15)],
                  3:[cx-bw*.12,shoulder],4:[cx-bw*.27,shoulder+remain*.04],5:[cx-bw*.43,shoulder],
                  6:[cx+bw*.12,shoulder],7:[cx+bw*.27,shoulder+remain*.04],8:[cx+bw*.43,shoulder],
                  9:[cx-bw*.08,hip],10:[cx-bw*.10,hip+remain*.22],11:[cx-bw*.13,y1-bh*.03],
                  12:[cx+bw*.08,hip],13:[cx+bw*.10,hip+remain*.22],14:[cx+bw*.13,y1-bh*.03]}
        for idx,p in adaptive.items():joints[idx]=nearest_point(foreground,p).astype(float).tolist()
    quality=[.3]*15;used=set();notes=[];routes=[]
    for n,(a,b,c) in enumerate(CHAINS):
        left=n%2==0;leg=n>=2;candidates=[]
        for i,endpoint in enumerate(endpoints):
            if i in used:continue
            x,y=endpoint['tip'];side=(cx-x) if left else (x-cx)
            if side<bw*.035:continue
            if leg:
                if y<cy+bh*.15:continue
                score=(y-cy)/bh+.15*endpoint['length']/bh-.12*abs(x-cx)/bw
            else:
                if y>cy+bh*.20:continue
                if neck is not None and y<neck+bh*.025:continue
                score=side/bw+.12*endpoint['length']/bh-.15*max(0,y-cy)/bh
                if neck is not None:score-=.9*abs(y-(neck+(y1-neck)*.14))/bh
            if x<2 or x>w-3 or y<2 or y>h-3:score-=.65
            candidates.append((score,i,endpoint))
        if not candidates:
            notes.append(f'{"Picior" if leg else "Braț"} {"stânga" if left else "dreapta"}: traseu ambiguu; verifică reperele.')
            continue
        _,i,endpoint=max(candidates,key=lambda item:item[0]);used.add(i)
        route=np.array(endpoint['route'],dtype=float) # extremitate → centru
        threshold=max(2.,min(float(dt.max())*.19,local_radius*.7))
        radii=dt[route[:,1].astype(int),route[:,0].astype(int)]
        eligible=radii>=threshold
        if leg:eligible&=route[:,1]<cy+bh*.23
        if neck is not None:eligible&=route[:,1]>=neck
        enters=np.flatnonzero(eligible)
        cut=int(enters[0]) if len(enters) else max(2,len(route)*2//3)
        cut=max(2,min(cut,len(route)-1))
        limb=route[:cut+1][::-1] # atașare → extremitate
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(limb,axis=0),axis=1))]
        if arc[-1]<bh*.09 or (not leg and limb[0,1]>cy+bh*.15):
            notes.append(f'{"Picior" if leg else "Braț"} {"stânga" if left else "dreapta"}: geometrie neplauzibilă; folosește corecția manuală.')
            continue
        midpoint=int(np.argmin(abs(arc-arc[-1]*.52)))
        bend=0.
        for k in range(2,len(limb)-2):
            if not .27<=arc[k]/arc[-1]<=.73:continue
            i0=int(np.argmin(abs(arc-(arc[k]-.17*arc[-1]))))
            i1=int(np.argmin(abs(arc-(arc[k]+.17*arc[-1]))))
            u=limb[k]-limb[i0];v=limb[i1]-limb[k]
            cosine=float(u@v)/max(float(np.linalg.norm(u)*np.linalg.norm(v)),1.e-8)
            angle=math.acos(np.clip(cosine,-1,1))
            if angle>bend+.02:bend=angle;midpoint=k
        joints[a]=limb[0].tolist();joints[b]=limb[midpoint].tolist();joints[c]=limb[-1].tolist()
        quality[a]=.78;quality[b]=.72 if bend>.25 else .42;quality[c]=.82
        if bend<=.25:notes.append(f'{"Genunchiul" if leg else "Cotul"} {"stânga" if left else "dreapta"}: membru aproape drept; poziția articulației este estimată pe traseu.')
        routes.append(limb.tolist())
    joints[0]=((np.array(joints[9])+joints[12])/2).tolist()
    joints[1]=((np.array(joints[3])+joints[6])/2).tolist()
    top=float(wy.min()) if len(wy) else y0
    joints[2]=nearest_point(foreground,[cx,top+.22*max(cy-top,1)]).astype(float).tolist()
    quality[0]=min(quality[9],quality[12]);quality[1]=min(quality[3],quality[6]);quality[2]=.45
    return np.round(joints,3).tolist(),dict(distance=dt,skeleton=skeleton,wide=wide,
        root=root.tolist(),neck_y=neck,endpoints=len(endpoints),routes=routes,quality=quality,notes=notes,
        method='Distanță chamfer + schelet Zhang–Suen + trasee geodezice; euristică, fără model neuronal')


def seed_line(labels,mask,a,b,label,steps=25):
    yy,xx=np.where(mask);points=np.stack([xx,yy],axis=1)
    if not len(points):return []
    seeds=[]
    for t in np.linspace(.25,.95,steps):
        p=np.array(a)*(1-t)+np.array(b)*t
        x,y=np.rint(p).astype(int)
        if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]):
            x,y=nearest_point(points,p)
        if labels[y,x] in (-1,label):labels[y,x]=label;seeds.append((int(y),int(x)))
    return seeds


def torso_seed(mask,joints):
    """Regiune anatomică propusă de umeri și șolduri, intersectată cu masca.

    Capul este inclus de partiționarea existentă. Nu este segmentare semantică
    neuronală: reperele sunt neuronale, regiunea corpului este geometrică.
    """
    j=np.asarray(joints);polygon=j[[3,6,12,9]]
    x,y=polygon.T
    area=abs(float(x@np.roll(y,1)-y@np.roll(x,1)))/2
    if area<max(9,int(mask.sum())*.002):return np.zeros_like(mask),False
    raster=Image.new('L',(mask.shape[1],mask.shape[0]),0)
    ImageDraw.Draw(raster).polygon([tuple(p) for p in polygon],fill=255)
    return (np.asarray(raster)>0)&mask,True


def partition(mask,joints,dt,body_override=None,labels_override=None,anatomical=False):
    h,w=mask.shape;j=np.array(joints)
    if labels_override is not None:
        labels=labels_override.copy();labels[~mask]=-1
        return labels,labels==0
    yy,xx=np.where(mask)
    labels=np.full((h,w),-1,dtype='int16');queue=deque()
    if body_override is None:
        # Nucleu gros; exclude vecinătățile segmentelor de membre, nu doar distanța în aer.
        body=mask&(dt>=max(2.,float(dt.max())*.26))
        for a,b,c in CHAINS:
            for u,v in ((a,b),(b,c)):
                vec=j[v]-j[u];den=max(float(vec@vec),1.e-6)
                t=np.clip(((np.stack([xx,yy],axis=1)-j[u])@vec)/den,0,1)
                d=np.linalg.norm(np.stack([xx,yy],axis=1)-(j[u]+t[:,None]*vec),axis=1)
                erase=(d<np.maximum(2,dt[yy,xx]*.9))&(t>.18)
                body[yy[erase],xx[erase]]=False
        if anatomical:
            torso,_=torso_seed(mask,joints)
            body|=torso
        if not body.any():
            idx=int(np.argmax(dt));body.flat[idx]=True
    else:body=body_override&mask
    labels[body]=0
    # Sursele corpului sunt introduse după sursele membrelor pentru egalități stabile.
    for n,(a,b,c) in enumerate(CHAINS):
        queue.extend(seed_line(labels,mask,j[a],j[b],1+2*n))
        queue.extend(seed_line(labels,mask,j[b],j[c],2+2*n))
    queue.extend((int(y),int(x)) for y,x in zip(*np.where(body)))
    while queue:
        y,x=queue.popleft();label=labels[y,x]
        for dy,dx in NEIGHBORS:
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and mask[ny,nx] and labels[ny,nx]==-1:
                if label==0 and body_override is not None:continue
                labels[ny,nx]=label;queue.append((ny,nx))
    # Fragmente fără surse rămân în corp, pentru a nu pierde pixeli ai desenului.
    missing=mask&(labels<0)
    if body_override is not None and missing.any():
        my,mx=np.where(missing);points=np.stack([mx,my],axis=1);distances=[]
        for a,b,c in CHAINS:
            for u,v in ((a,b),(b,c)):
                vec=j[v]-j[u];t=np.clip(((points-j[u])@vec)/max(float(vec@vec),1.e-8),0,1)
                distances.append(np.linalg.norm(points-(j[u]+t[:,None]*vec),axis=1))
        labels[my,mx]=np.argmin(np.stack(distances,axis=1),axis=1)+1
    else:labels[missing]=0
    return labels,labels==0


def structure(mask,joints,body_override=None,labels_override=None,geometry=None,anatomical=False):
    if geometry is None:
        dt=distance_transform(mask);skeleton=thin(mask)
    else:dt=geometry['distance'];skeleton=geometry['skeleton']
    labels,body=partition(mask,joints,dt,body_override,labels_override,anatomical)
    yy,xx=np.where(mask);points=np.stack([xx,yy],axis=1)
    checks=[]
    for i,p in enumerate(joints):
        x,y=np.rint(p).astype(int)
        on=0<=y<mask.shape[0] and 0<=x<mask.shape[1] and bool(mask[y,x])
        snap=nearest_point(points,p)
        checks.append(dict(joint=i,on_mask=on,distance_to_mask=round(float(np.linalg.norm(snap-p)),2)))
    return dict(distance=dt,skeleton=skeleton,labels=labels,body=body,checks=checks,
                manual_body=body_override is not None,manual_parts=labels_override is not None)


def diagnostic_images(texture: Image.Image,s:dict):
    mask=np.asarray(texture)[:,:,3]>0;dt=s['distance'];labels=s['labels'];body=s['body']
    scaled=np.clip(dt/max(float(dt.max()),1),0,1)
    heat=np.stack([30+scaled*215,65+scaled*140,100-scaled*60],axis=2).astype('uint8');heat[~mask]=[245,247,240]
    palette=np.array([p['color'] for p in PARTS],dtype='uint8')
    parts=np.full((*mask.shape,3),245,dtype='uint8');parts[mask]=palette[labels[mask]]
    overlay=np.array(texture.convert('RGB')).astype(float)
    overlay[body]=overlay[body]*.58+np.array([30,165,112])*.42
    sk=np.full((*mask.shape,3),245,dtype='uint8');sk[mask]=[215,226,210];sk[s['skeleton']]=[24,112,88]
    return {'distance':Image.fromarray(heat),'body_mask':Image.fromarray(body.astype('uint8')*255),
            'body_overlay':Image.fromarray(overlay.astype('uint8')),'parts':Image.fromarray(parts),
            'skeleton_map':Image.fromarray(sk),'part_labels':Image.fromarray(np.where(mask,labels,255).astype('uint8'))}


def extend_labels(labels,iterations=4):
    out=labels.copy()
    for _ in range(iterations):
        before=out.copy();p=np.pad(before,1,constant_values=-1)
        for dy,dx in NEIGHBORS:
            shifted=p[1+dy:1+dy+out.shape[0],1+dx:1+dx+out.shape[1]]
            take=(out<0)&(shifted>=0);out[take]=shifted[take]
    return out


def bind_protected(mesh,s):
    p=np.array(mesh['vertices']);j=np.array(mesh['joints']);labels=extend_labels(s['labels'])
    x=np.clip(np.rint(p[:,0]).astype(int),0,labels.shape[1]-1);y=np.clip(np.rint(p[:,1]).astype(int),0,labels.shape[0]-1)
    groups=np.maximum(labels[y,x],0);weights=[]
    for i,group in enumerate(groups):
        if group==0:weights.append([[0,1.]]);continue
        allowed=PARTS[int(group)]['chain'];dist=[]
        for bone in allowed:
            a,b=mesh['bones'][bone];v=j[b]-j[a];t=np.clip(float((p[i]-j[a])@v)/max(float(v@v),1.e-8),0,1)
            dist.append(float(np.linalg.norm(p[i]-(j[a]+t*v))))
        raw=1/np.maximum(dist,2.)**4;raw/=raw.sum()
        weights.append([[int(b),float(v)] for b,v in zip(allowed,raw)])
    mesh['protected_weights']=weights;mesh['vertex_parts']=groups.astype(int).tolist()
    mesh['parts']=PARTS
    mesh['body_vertex_indices']=np.flatnonzero(groups==0).tolist()
    mesh['rigid_weights']=[[[PARTS[int(group)]['bone'],1.]] for group in groups]
    return mesh


def puppet_bridges(texture:Image.Image,s,joints):
    """Porturi de legătură din frontierele reale ale pieselor, nu din proporții.

    Coordonatele și culorile rămân în repaus; rendererul transformă separat
    fiecare capăt. Frontiera este păstrată pentru filtrarea locală a îmbinării.
    """
    rgba=np.asarray(texture.convert('RGBA'));labels=s['labels'];j=np.asarray(joints,float)
    mask=rgba[:,:,3]>0
    dt=s['distance'] if 'distance' in s else distance_transform(mask)
    luminance=rgba[:,:,:3]@np.array([.2126,.7152,.0722])
    pixels={pid:np.column_stack(np.where((labels==pid)&(rgba[:,:,3]>0)))[:,::-1] for pid in range(9)}
    bridges=[]
    def unit(vector):return vector/max(float(np.linalg.norm(vector)),1.e-8)
    def outline_style(parent,child,ends,width):
        """Contrast la siluetă, confirmat prin mai multe profile locale.

        O umplere întunecată uniformă nu este un contur. Comparăm banda
        exterioară cu interiorul aceleiași piese, apoi estimăm grosimea
        cernelii din distanța ei la exterior. Evităm cusăturile etichetelor.
        """
        profiles=[];radius=max(5.,min(24.,width*.6))
        for end in ends:
            for pid in [parent,child]:
                points=pixels[pid];points=points[((points-end)**2).sum(axis=1)<=radius**2]
                if len(points)<8:continue
                x,y=points.T;depth=dt[y,x];light=luminance[y,x]
                edge=depth<=1.5;inside=depth>=max(2.,min(width*.3,radius*.8))
                if edge.sum()<3 or inside.sum()<3:continue
                dark=float(np.percentile(light[edge],20));fill=float(np.percentile(light[inside],65))
                contrast=fill-dark
                if contrast<35:continue
                threshold=dark+contrast*.45
                coverage=float((light[edge]<threshold).mean())
                if coverage<.5 or (light[inside]<threshold).mean()>.3:continue
                ink=(light<threshold)&(depth<=max(2.,min(8.,width*.35,radius*.75)))
                if ink.sum()<3:continue
                thickness=float(np.percentile(depth[ink],90)-.25)
                rgb=np.median(rgba[y[ink],x[ink],:3],axis=0)
                profiles.append((rgb,thickness,coverage))
        if len(profiles)<2:return None
        center=np.median([p[0] for p in profiles],axis=0)
        profiles=[p for p in profiles if np.linalg.norm(p[0]-center)<55]
        if len(profiles)<2:return None
        return dict(color=np.rint(np.median([p[0] for p in profiles],axis=0)).astype(int).tolist(),
                    width=round(float(np.clip(np.median([p[1] for p in profiles]),.6,min(8.,max(.6,width*.3)))),3),
                    confidence=round(float(np.mean([p[2] for p in profiles])),3))
    def color(pid,point,radius,outline):
        points=pixels[pid];distance=((points-point)**2).sum(axis=1)
        nearby=distance<=radius**2
        if not nearby.any():nearby[np.argmin(distance)]=True
        points=points[nearby];local=distance[nearby];weights=np.exp(-(local-local.min())/max(radius**2,1))
        values=rgba[points[:,1],points[:,0]];weights*=values[:,3]/255
        if outline:
            # La un desen conturat, cerneala nu trebuie să înnegrească umplutura.
            fill=luminance[points[:,1],points[:,0]]>np.dot(outline['color'],[.2126,.7152,.0722])+25
            if fill.any():weights*=fill
        return np.rint(np.average(values[:,:3],axis=0,weights=weights)).astype(int).tolist()
    for n,(root,middle,end) in enumerate(CHAINS):
        upper,lower=1+2*n,2+2*n
        for parent,child,joint,da,db in [
            (0,upper,root,j[middle]-j[root],j[middle]-j[root]),
            (upper,lower,middle,j[middle]-j[root],j[end]-j[middle])]:
            if not len(pixels[parent]) or not len(pixels[child]) or min(np.linalg.norm(da),np.linalg.norm(db))<1.e-6:continue
            edges=[]
            for a,b,offset in [(labels[:,:-1],labels[:,1:],[.5,0]),(labels[:-1],labels[1:],[0,.5])]:
                yy,xx=np.where(((a==parent)&(b==child))|((a==child)&(b==parent)))
                if len(xx):edges.append(np.column_stack([xx,yy])+offset)
            if not edges:continue
            points=np.concatenate(edges)
            # Urmărim frontiera conexă cea mai apropiată de articulație.
            seed=int(np.linalg.norm(points-j[joint],axis=1).argmin())
            doubled=np.rint(points*2).astype(int);lookup={tuple(point):i for i,point in enumerate(doubled)}
            component={seed};queue=deque([seed])
            while queue:
                index=queue.popleft();x,y=doubled[index]
                for dx,dy in [(2,0),(-2,0),(0,2),(0,-2),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    other=lookup.get((x+dx,y+dy))
                    if other is not None and other not in component:component.add(other);queue.append(other)
            points=points[sorted(component)]
            a,b=unit(da),unit(db)
            # Un braț lipit de trunchi poate avea o frontieră comună lungă.
            # Folosim grosimea locală a membrului pentru a rămâne lângă joint,
            # fără să transformăm acea frontieră laterală într-o membrană.
            seam=points[np.linalg.norm(points-j[joint],axis=1).argmin()]
            normal=np.array([-b[1],b[0]]);relative=pixels[child]-seam
            along=relative@b;across=relative@normal;step=max(1.,min(3.,np.linalg.norm(db)*.04))
            widths=[]
            for offset in [step,step*2,step*3]:
                strip=(abs(along-offset)<=max(.75,step*.5))&(abs(across)<=max(4.,min(50.,np.linalg.norm(db)*.4)))
                if strip.any():widths.append(float(np.ptp(across[strip])+1))
            local_width=float(np.median(widths)) if widths else 2.
            # Păstrăm secțiunea transversală întreagă, inclusiv capătul mai
            # îndepărtat de reper. Limităm separat contactele longitudinale
            # (de ex. un braț lipit de trunchi), ca să nu creăm o membrană.
            relative=points-seam
            points=points[(abs(relative@b)<=max(4.,local_width*.9+2))&
                          (abs(relative@normal)<=max(4.,local_width*1.6+2))]
            center=points.mean(axis=0)
            if len(points)>1:
                _,vectors=np.linalg.eigh((points-center).T@(points-center));axis=vectors[:,-1]
            else:axis=np.array([-b[1],b[0]])
            projection=(points-center)@axis
            width=float(np.ptp(projection)+1.5)
            thickness=float(max(1.,min(width,local_width)))
            inset=min(3.,max(.75,width*.12))
            # Capetele urmăresc pixelii reali; o dreaptă PCA recentrată ar
            # scurta tăieturile curbate sau asimetrice.
            ends=np.stack([points[projection.argmin()]-axis*.75,points[projection.argmax()]+axis*.75])
            port_a=ends-a*inset;port_b=ends+b*inset
            # Anvelopa întregii tăieturi, cu o margine pentru colțurile pixelilor.
            cut_points=np.concatenate([points+offset for offset in [[-.75,-.75],[-.75,.75],[.75,-.75],[.75,.75]]])
            ordered=sorted(set(map(tuple,cut_points)))
            def half(values):
                hull=[]
                for point in values:
                    while len(hull)>1:
                        u=np.subtract(hull[-1],hull[-2]);v=np.subtract(point,hull[-1])
                        if u[0]*v[1]-u[1]*v[0]>0:break
                        hull.pop()
                    hull.append(point)
                return hull
            cut=half(ordered)[:-1]+half(ordered[::-1])[:-1]
            outline=outline_style(parent,child,ends,width)
            bridges.append(dict(joint=joint,parts=[parent,child],bones=[PARTS[parent]['bone'],PARTS[child]['bone']],
                a=port_a.round(4).tolist(),b=port_b.round(4).tolist(),directions=[a.round(6).tolist(),b.round(6).tolist()],
                seam=points.round(4).tolist(),cut=np.round(cut,4).tolist(),thickness=round(thickness,4),
                outline=outline,
                colors=[color(parent,port_a.mean(axis=0),max(2,width*.45),outline),color(child,port_b.mean(axis=0),max(2,width*.45),outline)],
                width=round(width,4)))
    # Dacă mai multe articulații de pe membre diferite confirmă același stil,
    # îl putem continua și unde tăietura se află departe de silueta inițială.
    # Nu inventăm un contur global dintr-un singur detaliu întunecat.
    styled=[bridge for bridge in bridges if bridge['outline']]
    if len(styled)>=3 and len({(b['parts'][1]-1)//2 for b in styled})>=2:
        median=np.median([b['outline']['color'] for b in styled],axis=0)
        compatible=[b['outline'] for b in styled if np.linalg.norm(np.array(b['outline']['color'])-median)<55]
        if len(compatible)>=max(3,len(styled)*.75):
            shared=dict(color=np.rint(np.median([v['color'] for v in compatible],axis=0)).astype(int).tolist(),
                        width=round(float(np.median([v['width'] for v in compatible])),3),
                        confidence=round(float(np.mean([v['confidence'] for v in compatible])),3))
            for bridge in bridges:
                if bridge['outline'] is None:
                    bridge['outline']={**shared,'width':round(min(shared['width'],max(.6,bridge['width']*.3)),3)}
                    bridge['colors']=[color(pid,np.mean(bridge[key],axis=0),max(2,bridge['width']*.45),bridge['outline'])
                                      for pid,key in zip(bridge['parts'],['a','b'])]
    for bridge in bridges:
        bridge['ink']=[]
        if not bridge['outline']:continue
        ends=(np.array(bridge['a'])+np.array(bridge['b']))*.5
        radius=max(3.,bridge['thickness']*.7)
        ink_light=np.dot(bridge['outline']['color'],[.2126,.7152,.0722])
        for pid in bridge['parts']:
            points=pixels[pid];x,y=points.T
            take=(luminance[y,x]<ink_light+65)&(np.min(((points[:,None]-ends)**2).sum(axis=2),axis=1)<=radius**2)
            spans=[]
            for x,y in points[take]:
                if spans and spans[-1][1]==int(y) and spans[-1][0]+spans[-1][2]==int(x):spans[-1][2]+=1
                else:spans.append([int(x),int(y),1])
            bridge['ink'].append(spans)
    return bridges


def puppet_images(texture:Image.Image,s,joints):
    """Fiecare pixel aparține unei singure piese; racordurile închid rosturile.

    Capacele copiate din piesa vecină produceau pinteni la îndoiri inverse.
    Păstrăm numai proprietarul pixelului, inclusiv la umeri și șolduri.
    """
    alpha=np.asarray(texture)[:,:,3]>0;labels=s['labels']
    images={};spec=[]
    for part in PARTS:
        pid=part['id'];area=(labels==pid)&alpha;yy,xx=np.where(area)
        bounds=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
        piece=texture.copy();piece.putalpha(Image.fromarray(area.astype('uint8')*255))
        key=f'piece_{pid}';images[key]=piece
        spec.append({**part,'image_key':key,'pixels':int(area.sum()),'bounds':bounds})
    limbs=texture.copy();limbs.putalpha(Image.fromarray((alpha&(labels!=0)).astype('uint8')*255));images['limbs_texture']=limbs
    return images,spec
