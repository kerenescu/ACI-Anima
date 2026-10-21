// Date CMU reale, animații procedurale pentru comparație și LBS.
import {isMocap,animateMocapJoints} from './mocap.js';
// Ordinea părinților este topologică: orice părinte precede copilul.
export const MOTIONS = {
  wave: 'Procedural · Salut',
  walk: 'Procedural · Mers',
  dance: 'Procedural · Dans',
};
export const RENDER_MODES={puppet:'Marionetă · piese rigide',protected:'Corp rigid · membre flexibile',baseline:'LBS clasic · comparație'};
const BODY_JOINTS=new Set([0,1,2,3,6,9,12]);

export function animateJoints(rig, time, motion = 'wave', amplitude = 1, mode='protected',projection='auto') {
  if(isMocap(motion))return animateMocapJoints(rig,time,motion,amplitude,projection);
  const rest = rig.joints;
  const p = rest.map(v => [...v]);
  const phase = Math.PI * time; // perioadă 2 secunde
  const angle = Array(15).fill(0);
  let rootY = 0;
  if (motion === 'wave') {
    angle[7] = -.20 * amplitude;
    angle[8] = (.12 + .38 * Math.sin(phase * 2)) * amplitude;
    angle[2] = .035 * Math.sin(phase) * amplitude;
  } else if (motion === 'walk') {
    angle[10] = .28 * Math.sin(phase) * amplitude;
    angle[13] = -.28 * Math.sin(phase) * amplitude;
    angle[11] = .22 * Math.max(0, Math.sin(phase)) * amplitude;
    angle[14] = .22 * Math.max(0, -Math.sin(phase)) * amplitude;
    angle[4] = -.17 * Math.sin(phase) * amplitude;
    angle[7] = .17 * Math.sin(phase) * amplitude;
    rootY = -Math.abs(Math.sin(phase)) * 5 * amplitude;
  } else {
    angle[1] = .08 * Math.sin(phase) * amplitude;
    angle[4] = .24 * Math.sin(phase) * amplitude;
    angle[7] = -.24 * Math.sin(phase) * amplitude;
    angle[5] = .32 * Math.sin(phase * 2) * amplitude;
    angle[8] = -.32 * Math.sin(phase * 2) * amplitude;
    angle[10] = .15 * Math.sin(phase) * amplitude;
    angle[13] = -.15 * Math.sin(phase) * amplitude;
    rootY = -Math.abs(Math.sin(phase)) * 8 * amplitude;
  }
  p[0][1] += rootY;
  const accumulated = Array(15).fill(0);
  for (let i = 1; i < rest.length; i++) {
    const parent = rig.parents[i];
    accumulated[i] = accumulated[parent] + angle[i];
    const a = accumulated[i], c = Math.cos(a), s = Math.sin(a);
    const dx = rest[i][0] - rest[parent][0], dy = rest[i][1] - rest[parent][1];
    p[i] = [p[parent][0] + c * dx - s * dy, p[parent][1] + s * dx + c * dy];
  }
  if(mode==='baseline')return p;
  // Un singur reper rigid pentru bazin, torace, cap, umeri și șolduri.
  const bodyAngle=angle[1],c=Math.cos(bodyAngle),s=Math.sin(bodyAngle);
  for(const i of BODY_JOINTS){
    const dx=rest[i][0]-rest[0][0],dy=rest[i][1]-rest[0][1];
    p[i]=[rest[0][0]+c*dx-s*dy,rest[0][1]+rootY+s*dx+c*dy];
    accumulated[i]=bodyAngle;
  }
  for(let i=1;i<rest.length;i++){
    if(BODY_JOINTS.has(i))continue;
    const parent=rig.parents[i];accumulated[i]=accumulated[parent]+angle[i];
    const a=accumulated[i],c=Math.cos(a),s=Math.sin(a);
    const dx=rest[i][0]-rest[parent][0],dy=rest[i][1]-rest[parent][1];
    p[i]=[p[parent][0]+c*dx-s*dy,p[parent][1]+s*dx+c*dy];
  }
  return p;
}

export function boneTransforms(rig,target){
  return rig.bones.map(([a,b]) => {
    const r = rig.joints, p = target;
    const theta = Math.atan2(p[b][1]-p[a][1],p[b][0]-p[a][0])
                - Math.atan2(r[b][1]-r[a][1],r[b][0]-r[a][0]);
    const c=Math.cos(theta),s=Math.sin(theta);
    return {c,s,from:r[a],to:p[a],tx:p[a][0]-c*r[a][0]+s*r[a][1],ty:p[a][1]-s*r[a][0]-c*r[a][1]};
  });
}
export function skinVertices(rig, target,mode='protected') {
  const transforms=boneTransforms(rig,target);
  const weights=mode==='puppet'?(rig.rigid_weights||rig.weights):mode==='protected'?(rig.protected_weights||rig.weights):rig.weights;
  return rig.vertices.map(([x,y],i) => {
    let nx=0,ny=0,total=0;
    for (const [bone,weight] of weights[i]) {
      const t=transforms[bone],dx=x-t.from[0],dy=y-t.from[1];
      nx+=weight*(t.to[0]+t.c*dx-t.s*dy);
      ny+=weight*(t.to[1]+t.s*dx+t.c*dy);total+=weight;
    }
    return [nx/total,ny/total];
  });
}

export function drawRigidPiece(ctx,image,transform){
  if(!image)return;
  ctx.save();ctx.transform(transform.c,transform.s,-transform.s,transform.c,transform.tx,transform.ty);ctx.drawImage(image,0,0);ctx.restore();
}

const bridgeSurfaces=new WeakMap();
const puppetSurfaces=new WeakMap();
const transformPoint=(t,[x,y])=>[t.c*x-t.s*y+t.tx,t.s*x+t.c*y+t.ty];
const offsetPoint=(p,d,k)=>[p[0]+d[0]*k,p[1]+d[1]*k];
const midpoint=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
const distance=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1]);
const mixPoint=(a,b,t)=>a.map((value,i)=>value+(b[i]-value)*t);

function surface(cache,key){
  let result=cache.get(key);
  if(!result){
    const make=()=>typeof OffscreenCanvas==='function'?new OffscreenCanvas(1,1):document.createElement('canvas');
    result={canvas:make(),blur:make(),mask:make()};
    for(const name of ['canvas','blur','mask'])result[name+'Context']=result[name].getContext('2d');
    cache.set(key,result);
  }
  return result;
}
function clearSurface(canvas,ctx,width,height){
  if(canvas.width<width)canvas.width=Math.ceil(width/32)*32;
  if(canvas.height<height)canvas.height=Math.ceil(height/32)*32;
  ctx.resetTransform();ctx.globalCompositeOperation='source-over';ctx.globalAlpha=1;ctx.filter='none';
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

function convexHull(points){
  const sorted=points.map(p=>[...p]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const half=values=>{const hull=[];for(const p of values){while(hull.length>1&&cross(hull.at(-2),hull.at(-1),p)<=1.e-8)hull.pop();hull.push(p);}return hull;};
  return [...half(sorted).slice(0,-1),...half([...sorted].reverse()).slice(0,-1)];
}

export function jointBridgeShape(bridge,transforms){
  const [ta,tb]=bridge.bones.map(bone=>transforms[bone]);
  if(!ta||!tb||['c','s','tx','ty'].every(key=>Math.abs(ta[key]-tb[key])<1.e-7))return null;
  const a=bridge.a.map(p=>transformPoint(ta,p)),b=bridge.b.map(p=>transformPoint(tb,p));
  const [da,db]=bridge.directions.map(([x,y],i)=>{const t=i?tb:ta;return [t.c*x-t.s*y,t.s*x+t.c*y];});
  const ca=midpoint(...a),cb=midpoint(...b),width=bridge.width;
  const thickness=bridge.thickness||width;
  const rotation=Math.abs(Math.atan2(ta.c*tb.s-ta.s*tb.c,ta.c*tb.c+ta.s*tb.s));
  const sourceSeam=bridge.seam||bridge.a;
  const center=sourceSeam.reduce((sum,p)=>sum.map((v,i)=>v+p[i]/sourceSeam.length),[0,0]);
  const separation=distance(transformPoint(ta,center),transformPoint(tb,center));
  // Capace cu grosime reală. În apropierea repausului se aplatizează pe
  // tăietură; la îndoire formează un volum rotunjit, fără schimbare la ±180°.
  const fullness=Math.sqrt(Math.min(1,rotation/(Math.PI*.5)+separation/Math.max(thickness,1)));
  const capDepth=Math.min(thickness*.25,separation*.3+thickness*.08)*fullness;
  const cloud=[...a,...b];
  if(bridge.cut)for(const t of [ta,tb])cloud.push(...bridge.cut.map(p=>transformPoint(t,p)));
  for(const port of [a,b]){
    const middle=midpoint(...port),length=distance(...port)||1;
    const axis=port[1].map((v,i)=>(v-port[0][i])/length),normal=[-axis[1],axis[0]];
    for(let i=0;i<32;i++){
      const angle=2*Math.PI*i/32,u=Math.cos(angle)*thickness*.5,v=Math.sin(angle)*capDepth;
      cloud.push([middle[0]+axis[0]*u+normal[0]*v,middle[1]+axis[1]*u+normal[1]*v]);
    }
  }
  // Anvelopa convexă nu depinde de împerecherea capetelor. Rămâne simplă
  // inclusiv la ±180°, când apropierea capetelor ar inversa cele două laturi.
  // Mânerele Bézier rămân în anvelopă: nu apar bucle sau pinteni exteriori.
  const hull=convexHull(cloud);if(hull.length<3)return null;
  const corners=hull.map((p,i)=>{
    const before=hull[(i+hull.length-1)%hull.length],after=hull[(i+1)%hull.length];
    const radius=Math.min(thickness*.08,distance(p,before)*.18,distance(p,after)*.18);
    return {p,entry:mixPoint(p,before,radius/distance(p,before)),exit:mixPoint(p,after,radius/distance(p,after))};
  });
  const curves=[];
  corners.forEach(({p,entry,exit},i)=>{
    // Conversie exactă a unui arc pătratic în Bézier cubic, apoi latură dreaptă.
    curves.push([mixPoint(entry,p,2/3),mixPoint(exit,p,2/3),exit]);
    const next=corners[(i+1)%corners.length].entry;
    curves.push([mixPoint(exit,next,1/3),mixPoint(exit,next,2/3),next]);
  });
  // Conturul este perimetrul volumului, desenat sub ambele piese originale.
  // Nu există margini transversale în interiorul acestui volum continuu.
  const contour=[{start:corners[0].entry,curves}];
  const seams=[ta,tb].map(t=>(bridge.seam||bridge.a).map(p=>transformPoint(t,p)));
  const points=[...hull,...seams.flat()];
  if(!points.every(p=>p.every(Number.isFinite)))return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const strength=Math.min(1,rotation/.2+separation/Math.max(width,1));
  const sigma=Math.max(.45,Math.min(1.6,width*.045));
  const margin=6*sigma+2;
  return {start:corners[0].entry,curves,contour,hull,seams,sigma,strength:strength*strength*(3-2*strength),
    from:offsetPoint(ca,da,-width*.2),to:offsetPoint(cb,db,width*.2),
    bounds:{x:Math.floor(Math.min(...xs)-margin),y:Math.floor(Math.min(...ys)-margin),
      right:Math.ceil(Math.max(...xs)+margin),bottom:Math.ceil(Math.max(...ys)+margin)}};
}

function drawJointBridge(context,bridge,shape,outline=true){
  const [a,b]=bridge.colors,mean=a.map((value,i)=>(value+b[i])/2);
  if(Math.hypot(shape.to[0]-shape.from[0],shape.to[1]-shape.from[1])>1.e-6){
    const gradient=context.createLinearGradient(...shape.from,...shape.to);
    gradient.addColorStop(0,`rgb(${a.join(',')})`);gradient.addColorStop(.5,`rgb(${mean.join(',')})`);gradient.addColorStop(1,`rgb(${b.join(',')})`);
    context.fillStyle=gradient;
  }else context.fillStyle=`rgb(${mean.join(',')})`;
  context.beginPath();context.moveTo(...shape.start);
  for(const curve of shape.curves)context.bezierCurveTo(...curve.flat());
  context.closePath();context.fill();
  if(outline&&bridge.outline){
    context.save();context.clip();
    context.strokeStyle=`rgb(${bridge.outline.color.join(',')})`;
    // Contur în interiorul siluetei, cu aceeași grosime ca cerneala din sursă.
    context.lineWidth=bridge.outline.width*2;context.lineCap='round';context.lineJoin='round';
    context.beginPath();
    for(const rail of shape.contour){context.moveTo(...rail.start);for(const curve of rail.curves)context.bezierCurveTo(...curve.flat());}
    context.stroke();context.restore();
  }
}

function drawPuppetLayers(ctx,rig,transforms,getImage,order,bridges){
  for(const id of order){
    // Racordul se află sub ambele piese; niciun decupaj alpha dublu la contact.
    for(const {bridge,shape} of bridges){
      if(order[Math.min(...bridge.parts.map(part=>order.indexOf(part)))]===id)drawJointBridge(ctx,bridge,shape);
    }
    const p=rig.puppet_parts?.find(p=>p.id===id);if(p)drawRigidPiece(ctx,getImage(p.image_key),transforms[p.bone]);
  }
}

function featherJoint(frame,frameBounds,density,bridge,shape,rig,transforms,getImage){
  const {x,y,right,bottom}=shape.bounds;
  const width=Math.ceil((right-x)*density),height=Math.ceil((bottom-y)*density);
  if(width<1||height<1||width*height>2_000_000)return;
  const local=surface(bridgeSurfaces,bridge);
  const {canvas,canvasContext:c,blur,blurContext:b,mask,maskContext:m}=local;
  for(const [image,context] of [[canvas,c],[blur,b],[mask,m]])clearSurface(image,context,width,height);
  // Filtrăm compozitul real, inclusiv culoarea piesei, nu doar umplutura.
  // Canvas aplică un nucleu Gaussian separabil pe culori premultiplicate alpha.
  const sx=(x-frameBounds.x)*density,sy=(y-frameBounds.y)*density;
  c.drawImage(frame,sx,sy,width,height,0,0,width,height);
  if(bridge.outline&&bridge.ink?.some(spans=>spans.length)){
    if(local.inkSource!==bridge.ink){
      local.inkSource=bridge.ink;local.inkPaths=bridge.ink.map(spans=>{const path=new Path2D();for(const [x,y,length] of spans)path.rect(x,y,length,1);return path;});
    }
    m.setTransform(density,0,0,density,-x*density,-y*density);m.fillStyle='white';m.strokeStyle='white';m.lineWidth=1;
    local.inkPaths.forEach((path,i)=>{const t=transforms[bridge.bones[i]];m.save();m.transform(t.c,t.s,-t.s,t.c,t.tx,t.ty);m.fill(path);m.stroke(path);m.restore();});
    // Numai cerneala veche ajunsă în interiorul volumului poate fi acoperită.
    // Păstrăm o bandă de protecție de-a lungul siluetei vizibile.
    m.resetTransform();m.globalCompositeOperation='destination-in';
    const clearance=(bridge.outline.width+1)*density;
    for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1],[-.7,-.7],[-.7,.7],[.7,-.7],[.7,.7]])m.drawImage(canvas,dx*clearance,dy*clearance);
    b.setTransform(density,0,0,density,-x*density,-y*density);b.fillStyle='white';b.beginPath();b.moveTo(...shape.start);
    for(const curve of shape.curves)b.bezierCurveTo(...curve.flat());b.closePath();b.fill();
    m.drawImage(blur,0,0);m.setTransform(density,0,0,density,-x*density,-y*density);m.globalCompositeOperation='destination-out';
    for(const part of rig.puppet_parts||[])if(!bridge.parts.includes(part.id))drawRigidPiece(m,getImage(part.image_key),transforms[part.bone]);
    clearSurface(blur,b,width,height);b.setTransform(density,0,0,density,-x*density,-y*density);drawJointBridge(b,bridge,shape,false);
    b.resetTransform();b.globalCompositeOperation='destination-in';b.drawImage(mask,0,0);b.globalCompositeOperation='source-over';
    c.save();c.globalAlpha=shape.strength;c.drawImage(blur,0,0);c.restore();
    const destination=frame.getContext('2d');destination.save();destination.resetTransform();destination.globalAlpha=shape.strength;destination.drawImage(blur,0,0,width,height,sx,sy,width,height);destination.restore();
    clearSurface(blur,b,width,height);clearSurface(mask,m,width,height);
  }
  b.filter=`blur(${shape.sigma*density}px)`;b.drawImage(canvas,0,0);b.filter='none';
  m.setTransform(density,0,0,density,-x*density,-y*density);
  m.filter=`blur(${shape.sigma*density*.55}px)`;m.fillStyle='white';
  m.beginPath();
  for(const seam of shape.seams)for(const [px,py] of seam){m.moveTo(px+shape.sigma*1.8,py);m.arc(px,py,shape.sigma*1.8,0,Math.PI*2);}
  m.fill();
  m.filter='none';m.resetTransform();
  // O eroziune alpha păstrează conturul exterior clar și nu întinde fundalul.
  m.globalCompositeOperation='destination-in';
  const erosion=Math.max(1,shape.sigma*density);
  for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1],[-.7,-.7],[-.7,.7],[.7,-.7],[.7,.7]])m.drawImage(canvas,dx*erosion,dy*erosion);
  // Piesele care nu participă la această articulație își păstrează detaliile.
  m.setTransform(density,0,0,density,-x*density,-y*density);m.globalCompositeOperation='destination-out';
  for(const part of rig.puppet_parts||[])if(!bridge.parts.includes(part.id))drawRigidPiece(m,getImage(part.image_key),transforms[part.bone]);
  b.globalCompositeOperation='destination-in';b.drawImage(mask,0,0);b.globalCompositeOperation='source-over';
  const destination=frame.getContext('2d');destination.save();destination.resetTransform();destination.globalAlpha=shape.strength;
  destination.drawImage(blur,0,0,width,height,sx,sy,width,height);destination.restore();
}

export function drawPuppet(ctx,rig,target,getImage,front=false,depthOrder=null,smoothJoints=true){
  const transforms=boneTransforms(rig,target);
  const order=front?[0,5,6,7,8,1,2,3,4]:(depthOrder||[5,6,7,8,1,2,3,4,0]);
  const bridges=smoothJoints?(rig.joint_bridges||[]).map(bridge=>({bridge,shape:jointBridgeShape(bridge,transforms)})).filter(item=>item.shape):[];
  if(!bridges.length){drawPuppetLayers(ctx,rig,transforms,getImage,order,[]);return;}
  const points=bridges.flatMap(({shape:{bounds:b}})=>[[b.x,b.y],[b.right,b.bottom]]);
  for(const part of rig.puppet_parts||[]){
    const image=getImage(part.image_key);if(!image||part.pixels===0)continue;
    const [x,y,right,bottom]=part.bounds||[0,0,image.width,image.height];
    points.push(...[[x,y],[right,y],[right,bottom],[x,bottom]].map(p=>transformPoint(transforms[part.bone],p)));
  }
  const x=Math.floor(Math.min(...points.map(p=>p[0])))-2,y=Math.floor(Math.min(...points.map(p=>p[1])))-2;
  const w=Math.ceil(Math.max(...points.map(p=>p[0])))-x+2,h=Math.ceil(Math.max(...points.map(p=>p[1])))-y+2;
  const matrix=ctx.getTransform(),scale=Math.max(Math.hypot(matrix.a,matrix.b),Math.hypot(matrix.c,matrix.d));
  const density=Math.min(Math.max(1,Math.min(4,scale)),Math.sqrt(8_000_000/(w*h)));
  const width=Math.ceil(w*density),height=Math.ceil(h*density);
  const {canvas,canvasContext:context}=surface(puppetSurfaces,rig);
  clearSurface(canvas,context,width,height);context.setTransform(density,0,0,density,-x*density,-y*density);
  drawPuppetLayers(context,rig,transforms,getImage,order,bridges);
  for(const {bridge,shape} of bridges)featherJoint(canvas,{x,y},density,bridge,shape,rig,transforms,getImage);
  ctx.drawImage(canvas,0,0,width,height,x,y,width/density,height/density);
}

export function measureDeformation(rig,target,mode='protected'){
  const moved=skinVertices(rig,target,mode);let body=0,limbs=0,count=0;
  const groups=rig.vertex_parts||rig.vertices.map(()=>0);
  for(const tri of rig.triangles||[]){
    for(const [a,b] of [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]]){
      if(groups[a]!==groups[b])continue;
      const before=Math.hypot(rig.vertices[a][0]-rig.vertices[b][0],rig.vertices[a][1]-rig.vertices[b][1]);
      if(before<1.e-7)continue;
      const after=Math.hypot(moved[a][0]-moved[b][0],moved[a][1]-moved[b][1]);
      const delta=Math.abs(after/before-1)*100;
      if(groups[a]===0){body=Math.max(body,delta);count++;}else limbs=Math.max(limbs,delta);
    }
  }
  return {bodyMaxPercent:body<1.e-8?0:body,limbMaxPercent:limbs<1.e-8?0:limbs,bodyEdgesChecked:count};
}

export function affineTriangle(ctx, image, src, dst) {
  const [[x0,y0],[x1,y1],[x2,y2]]=src;
  const [[u0,v0],[u1,v1],[u2,v2]]=dst;
  const dx1=x1-x0,dy1=y1-y0,dx2=x2-x0,dy2=y2-y0;
  const det=dx1*dy2-dx2*dy1;
  if(Math.abs(det)<1e-7)return;
  const a=((u1-u0)*dy2-(u2-u0)*dy1)/det;
  const c=(dx1*(u2-u0)-dx2*(u1-u0))/det;
  const b=((v1-v0)*dy2-(v2-v0)*dy1)/det;
  const d=(dx1*(v2-v0)-dx2*(v1-v0))/det;
  ctx.save();
  // O mică suprapunere acoperă rosturile de antialiasing dintre triunghiuri.
  const cx=(u0+u1+u2)/3,cy=(v0+v1+v2)/3;
  ctx.beginPath();dst.forEach(([x,y],i)=>{
    const dx=x-cx,dy=y-cy,len=Math.hypot(dx,dy)||1;
    const px=x+dx/len*.85,py=y+dy/len*.85;
    if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);
  });ctx.closePath();ctx.clip();
  ctx.transform(a,b,c,d,u0-a*x0-c*y0,v0-b*x0-d*y0);
  ctx.drawImage(image,0,0);ctx.restore();
}
