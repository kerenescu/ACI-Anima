// Redare din coordonate CMU înregistrate, fără sinusoide procedurale.
const clips=new Map();
const prepared=new WeakMap();
const boundsCache=new WeakMap();
const BODY=new Set([0,1,2,3,6,9,12]);
const LIMBS=[4,5,7,8,10,11,13,14];
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const isMocap=id=>typeof id==='string'&&id.startsWith('cmu_');

export function registerMocapClip(clip){
 if(clip.format!=='aci-cmu-motion'||!Array.isArray(clip.positions)||clip.positions.length<2||clip.fps<=0)throw Error('Înregistrare CMU invalidă.');
 if(clip.positions.length!==clip.frames||clip.root_y.length!==clip.frames)throw Error('Numărul cadrelor CMU nu corespunde.');
 if(!clip.positions.every(frame=>frame.length===15&&frame.every(p=>p.length===3&&p.every(Number.isFinite))))throw Error('Coordonate CMU invalide.');
 clips.set(clip.id,clip);return clip;
}
export const getMocapClip=id=>clips.get(id);
export function motionDuration(id){return clips.get(id)?.duration||2;}

export async function loadMocapClip(id){
 if(!isMocap(id))return null;
 if(clips.has(id))return clips.get(id);
 const response=await fetch('/api/motion/'+encodeURIComponent(id));
 if(!response.ok)throw Error('Înregistrarea CMU nu s-a încărcat. Verifică fișierul assets/motions.json.');
 return registerMocapClip(await response.json());
}

function preparedMotion(clip,projection){
 let cache=prepared.get(clip);if(!cache){cache=new Map();prepared.set(clip,cache);}if(cache.has(projection))return cache.get(projection);
 const axis=group=>projection==='front'?[1,0]:projection==='side'?[0,1]:clip.planes[group].axis;
 const angles=[],body=[];
 for(let f=0;f<clip.frames;f++){
  const p=clip.positions[f],row=Array(15).fill(0);
  for(const j of LIMBS){const parent=j-1,u=axis(j<9?'arms':'legs'),v=p[j].map((v,c)=>v-p[parent][c]);
   const x=v[0]*u[0]+v[2]*u[1],y=-v[1];
   let a=Math.hypot(x,y)>1.e-6?Math.atan2(y,x):(f?angles[f-1][j]:Math.PI/2);
   if(f)a=angles[f-1][j]+wrap(a-angles[f-1][j]);row[j]=a;
  }
  const dx=p[1][0]-p[0][0],dy=p[1][1]-p[0][1];let a=Math.atan2(dx,dy);
  if(f)a=body[f-1]+wrap(a-body[f-1]);body.push(a);angles.push(row);
 }
 const result={angles,body};cache.set(projection,result);return result;
}

export function sampleMocap(id,time){
 const clip=clips.get(id);if(!clip)throw Error('Mișcarea CMU trebuie încărcată înainte de redare.');
 const duration=clip.duration,t=((time%duration)+duration)%duration,f=t*clip.fps;
 const a=Math.min(Math.floor(f),clip.frames-1),b=Math.min(a+1,clip.frames-1),mix=f-a;
 const positions=clip.positions[a].map((p,j)=>p.map((v,c)=>v+(clip.positions[b][j][c]-v)*mix));
 return {clip,time:t,a,b,mix,positions,rootY:clip.root_y[a]+(clip.root_y[b]-clip.root_y[a])*mix};
}

export function animateMocapJoints(rig,time,id,amplitude=1,projection='auto'){
 if(amplitude===0)return rig.joints.map(p=>[...p]);
 const sample=sampleMocap(id,time),{clip,a,b,mix}=sample,data=preparedMotion(clip,projection);
 const lerp=values=>values[a]+(values[b]-values[a])*mix;
 const theta=lerp(data.body)*amplitude,c=Math.cos(theta),s=Math.sin(theta),rest=rig.joints,p=rest.map(v=>[...v]);
 const ys=rest.map(p=>p[1]),height=Math.max(...ys)-Math.min(...ys),rootY=-sample.rootY*height/Math.max(clip.source_height,1)*amplitude;
 const bodyReference=Math.atan2(rest[1][1]-rest[0][1],rest[1][0]-rest[0][0])+Math.PI/2;
 for(const j of BODY){const x=rest[j][0]-rest[0][0],y=rest[j][1]-rest[0][1];p[j]=[rest[0][0]+c*x-s*y,rest[0][1]+rootY+s*x+c*y];}
 for(const j of LIMBS){const parent=rig.parents[j],dx=rest[j][0]-rest[parent][0],dy=rest[j][1]-rest[parent][1];
  const length=Math.hypot(dx,dy),restAngle=Math.atan2(dy,dx);
  const source=data.angles[a][j]+(data.angles[b][j]-data.angles[a][j])*mix+bodyReference;
  const angle=restAngle+wrap(source-restAngle)*amplitude;
  p[j]=[p[parent][0]+length*Math.cos(angle),p[parent][1]+length*Math.sin(angle)];
 }
 return p;
}

export function mocapDepthOrder(id,time){
 const p=sampleMocap(id,time).positions;
 const parts=[[0,(p[0][2]+p[1][2])*.5]];
 for(const [part,a,b] of [[1,3,4],[2,4,5],[3,6,7],[4,7,8],[5,9,10],[6,10,11],[7,12,13],[8,13,14]])parts.push([part,(p[a][2]+p[b][2])*.5]);
 return parts.sort((a,b)=>a[1]-b[1]||a[0]-b[0]).map(item=>item[0]);
}

export function mocapBounds(rig,id,projection,width,height){
 if(!isMocap(id))return {x0:0,y0:0,x1:width,y1:height};
 let cached=boundsCache.get(rig);if(!cached){cached=new Map();boundsCache.set(rig,cached);}
 const key=id+':'+projection+':'+width+':'+height;if(cached.has(key))return cached.get(key);
 const clip=clips.get(id);let x0=0,y0=0,x1=width,y1=height;
 const radius=Math.hypot(width,height);
 for(let f=0;f<clip.frames;f++){
  const joints=animateMocapJoints(rig,f/clip.fps,id,1,projection);
  for(const [x,y] of joints){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}
  // Capul și conturul corpului pot depăși controalele anatomice. Urmărim
  // colțurile texturii rotite de transformarea rigidă a corpului.
  const theta=Math.atan2(joints[1][1]-joints[0][1],joints[1][0]-joints[0][0])-Math.atan2(rig.joints[1][1]-rig.joints[0][1],rig.joints[1][0]-rig.joints[0][0]);
  const c=Math.cos(theta),s=Math.sin(theta);
  for(const [x,y] of [[0,0],[width,0],[width,height],[0,height]]){
   const dx=x-rig.joints[0][0],dy=y-rig.joints[0][1],nx=joints[0][0]+c*dx-s*dy,ny=joints[0][1]+s*dx+c*dy;
   x0=Math.min(x0,nx);x1=Math.max(x1,nx);y0=Math.min(y0,ny);y1=Math.max(y1,ny);
  }
 }
 const pad=radius*.055,result={x0:x0-pad,y0:y0-pad,x1:x1+pad,y1:y1+pad};cached.set(key,result);return result;
}

export function drawMocapSource(canvas,id,time,parents){
 const sample=sampleMocap(id,time),clip=sample.clip,ctx=canvas.getContext('2d');
 const width=canvas.clientWidth||800,height=260,ratio=Math.min(globalThis.devicePixelRatio||1,2);
 if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);}
 ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
 const radius=clip.source_height*.85;
 for(const [panel,axis,label] of [[0,0,'FRONTAL · X / Y'],[1,2,'PROFIL · Z / Y']]){
  const pw=width/2,scale=Math.min((pw-30)/(radius*2),(height-50)/(clip.source_height*1.6));
  ctx.save();ctx.beginPath();ctx.rect(panel*pw,0,pw,height);ctx.clip();ctx.fillStyle='#176c58';ctx.font='600 11px Segoe UI';ctx.fillText(label,panel*pw+14,20);
  const points=sample.positions.map(p=>[panel*pw+pw/2+p[axis]*scale,height*.58-(p[1]+sample.rootY)*scale]);
  for(let j=1;j<points.length;j++){ctx.strokeStyle=j<6||j>=9&&j<12?'#d88450':'#16816b';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...points[parents[j]]);ctx.lineTo(...points[j]);ctx.stroke();}
  for(const [x,y] of points){ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fillStyle='#194c40';ctx.fill();}ctx.restore();
 }
}
