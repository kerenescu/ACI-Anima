import {STEPS, METHOD_HTML} from './content.js';
import {MOTIONS,RENDER_MODES,animateJoints,skinVertices,affineTriangle,boneTransforms,drawRigidPiece,drawPuppet,measureDeformation} from './motion.js';
import {isMocap,loadMocapClip,getMocapClip,motionDuration,mocapBounds,mocapDepthOrder,drawMocapSource} from './mocap.js';
import {cropRect,cropHandles,hitCrop,moveCrop,resizeCrop} from './crop-editor.js';

const $=id=>document.getElementById(id);
const MOTION_GROUP_ORDER=['emotes','dance','locomotion','styles','sports','procedural'];
let state=null,stage=0,tab='concept',view='original',busy=false;
let imageCache=new Map(),selectedJoint=4,drag=null,cropEdit=false;
let cropDraft=null,cropDirty=false,cropDrawMode=false;
let maskCanvas=null,maskDirty=false,maskEditing=false,brushMode='add',brushSize=10,undo=[];
let editorKind='mask',editorPreview=null,editorOriginalMask=null,partPaint=0,bodyBoxEdit=false;
let playing=true,motion='cmu_07_01',amplitude=1,showSkeleton=true,showMesh=true;
let motionProjection='auto',showMotionSource=true;
let motionCatalog=[],motionCategory='all';
let renderMode='puppet',comparison=false,limbsFront=false,smoothJoints=true,lastMetric=0;
let comparePose=false;
let animationTime=0,previousTick=performance.now(),camera=null,toastTimer;
const canvas=$('canvas');

function toast(message,error=false){
  $('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,error?6500:3500);
}
async function api(path,body){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!response.ok){const err=await response.json().catch(()=>({}));throw Error(err.error||'Nu am putut finaliza operația.');}
  return response.json();
}
function setBusy(value,label='Se calculează…'){
  busy=value;$('busyOverlay').hidden=!value;$('busyText').textContent=label;
  document.querySelectorAll('button').forEach(b=>{
    if(value){b.dataset.disabledBefore=String(b.disabled);b.disabled=true;}
    else if(b.dataset.disabledBefore!==undefined){b.disabled=b.dataset.disabledBefore==='true';delete b.dataset.disabledBefore;}
  });
  if(!value){updateButtons();updateNavigation();}
}
async function operation(fn,label){
  if(busy)return;setBusy(true,label);
  try{await fn();}catch(e){console.error(e);toast(e.message,true);}finally{setBusy(false);}
}
async function applyState(next){
  const oldId=state?.id;state=next;
  if(oldId!==state.id){imageCache.clear();maskCanvas=null;undo=[];maskDirty=false;maskEditing=false;comparePose=false;motionCategory='all';cropDraft=null;cropDirty=false;cropEdit=false;}
  if(oldId!==state.id){renderMode=state.settings?.render_mode||'puppet';motion=state.settings?.motion||'cmu_07_01';motionProjection=state.settings?.motion_projection||'auto';showMotionSource=state.settings?.show_motion_source!==false;comparison=!!state.settings?.comparison;limbsFront=!!state.settings?.limbs_front;smoothJoints=state.settings?.smooth_joints!==false;}
  await loadMocapClip(motion);
  await Promise.all(Object.entries(state.images).map(async([key,url])=>{
    const existing=imageCache.get(key);if(existing?.url===url)return;
    const img=new Image();await new Promise((ok,no)=>{img.onload=ok;img.onerror=()=>no(Error('O imagine intermediară nu s-a încărcat.'));img.src=url;});
    imageCache.set(key,{url,img});
  }));
}
const currentImage=key=>imageCache.get(key)?.img;

function updateButtons(){
  $('nextBtn').disabled=!state||busy||stage===9;
  $('prevBtn').disabled=stage===0||busy;
  $('runAllBtn').disabled=!state||busy;
  $('saveBtn').disabled=!state||busy;
  $('downloadImageBtn').disabled=!state||busy;
  $('nextBtn').textContent=stage<9?`${STEPS[stage+1].short} →`:'Parcurs încheiat ✓';
}
function updateNavigation(){
  $('steps').innerHTML='';
  STEPS.forEach((s,i)=>{
    const b=document.createElement('button');b.className='step-nav'+(i===stage?' active':'')+(state&&i<=state.computed?' done':'');
    b.dataset.step=i;b.title=s.title;b.setAttribute('aria-current',i===stage?'step':'false');
    b.innerHTML=`<span class="step-num">${state&&i<state.computed&&i!==stage?'✓':String(i+1).padStart(2,'0')}</span><span>${s.short}</span>`;
    b.disabled=busy||(i>0&&!state);b.onclick=()=>go(i);$('steps').append(b);
  });
  document.querySelectorAll('.example').forEach(b=>b.classList.toggle('selected',b.dataset.example===state?.example));
}
function renderLesson(){
  const s=STEPS[stage];$('lessonContent').innerHTML=s[tab];
  $('stepInput').textContent=s.input;$('stepOutput').textContent=s.output;
  document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===tab)));
}
function resetView(){
  view=STEPS[stage].key;cropEdit=false;cropDraft=null;cropDirty=false;bodyBoxEdit=false;maskEditing=false;maskCanvas=null;maskDirty=false;undo=[];editorKind='mask';editorPreview=null;
}
function updateUI(){
  const s=STEPS[stage];
  $('stageTitle').textContent=s.title;$('stageSubtitle').textContent=s.subtitle;
  $('stageTag').textContent=s.tag;$('stageCount').textContent=`ÉTAPE ${String(stage+1).padStart(2,'0')} / 10`.replace('ÉTAPE','PASUL');
  $('stageTime').textContent=s.time;$('emptyState').hidden=!!state;
  $('uploadSection').hidden=stage!==0;$('histogramSection').hidden=stage!==3;
  $('exports').hidden=stage!==9;$('progressText').textContent=state?`${Math.min(state.computed+1,10)} din 10 etape disponibile`:'Încarcă un desen pentru a începe';
  $('imageCaption').textContent=stage===6&&state?(view.startsWith('heatmap_')?`${state.joint_names[selectedJoint]} · răspuns neuronal${selectedJoint<2?' · media hărților perechii, reper calculat':''}`:comparePose?'Punctat violet: geometrie · continuu verde: repere curente · dezacordul nu este eroare măsurată':state.pose_kind):state?.name||'PNG, JPEG, WebP · un singur personaj';
  $('dimensions').textContent=state?(stage<=1?state.meta.source_size.join(' × '):(currentImage('crop')?[currentImage('crop').naturalWidth,currentImage('crop').naturalHeight].join(' × '):'—'))+' px':'—';
  if(stage===5&&state)$('imageCaption').textContent=state.name+' · '+(state.meta.segmentation?.description||'Segmentare geometrică')+(maskEditing?' · editezi selecția verde':'');
  if(stage===8&&isMocap(motion))$('imageCaption').textContent=`${MOTIONS[motion]} · înregistrare reală · GIF: fragment de 2 s din poziția curentă`;
  $('mocapSection').hidden=stage!==8||!isMocap(motion)||!showMotionSource;
  document.querySelector('.canvas-card').classList.toggle('threshold-card',stage===3);
  document.querySelector('.canvas-card').classList.toggle('comparing',stage===8&&comparison);
  updateNavigation();renderLesson();renderViewButtons();renderToolbar();renderMetrics();renderPartsLegend();updateButtons();draw();
  $('warnings').innerHTML='';
  if(stage>=5)for(const warning of state?.meta.warnings||[]){const p=document.createElement('p');p.textContent=warning;$('warnings').append(p);}
  if(stage>=6)for(const warning of state?.meta.pose_diagnostics?.notes||[]){const p=document.createElement('p');p.textContent=warning;$('warnings').append(p);}
  if(stage===3)drawHistogram();
}
const VIEWS={
  0:[['original','Original']],1:[['original','Dreptunghi'],['crop','Decupaj']],
  2:[['gray','Intensitate'],['background','Fundal'],['corrected','Corectată']],
  3:[['corrected','Intensități'],['ink','Urme selectate']],
  4:[['ink','Înainte'],['closed','După închidere']],
  5:[['overlay','Selecție'],['mask','Mască'],['filled','Umplere'],['texture','Textură']],
  6:[['crop','Repere'],['distance','Distanță'],['skeleton_map','Schelet'],['body_overlay','Corp'],['parts','Zone']],
  7:[['texture','Suprafață'],['parts','Piese']],8:[['texture','Animație']],9:[['texture','Rezultat']],
};
function renderViewButtons(){
  $('viewButtons').innerHTML='';
  const choices=[...VIEWS[stage]];
  if(stage===6&&state?.images?.neural_input)choices.push([`heatmap_${selectedJoint}`,'Răspuns neuronal'],['neural_input','Intrarea rețelei']);
  for(const [key,label] of choices){
    const b=document.createElement('button');b.textContent=label;b.className=view===key?'active':'';
    b.disabled=(stage===5&&maskEditing&&editorKind==='mask'&&key!=='overlay')||(stage===1&&cropEdit&&key!=='original');
    b.onclick=()=>{if(maskEditing){if(key!==view)toast('Aplică sau anulează editarea înainte de a schimba vederea.');return;}view=key;updateUI();};
    $('viewButtons').append(b);
  }
  $('visualTitle').textContent=stage===0?'IMAGINEA ORIGINALĂ':stage===6?'REPERE EDITABILE':stage>=7?'PERSONAJUL NOSTRU':STEPS[stage].tag;
}
function toolbarButton(label,fn,active=false){
  const b=document.createElement('button');b.textContent=label;if(active)b.className='active';b.onclick=fn;$('editToolbar').append(b);return b;
}
function checkbox(label,checked,onchange){
  const l=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.checked=checked;c.onchange=()=>onchange(c.checked);l.append(c,document.createTextNode(label));$('editToolbar').append(l);
}
function motionChoices(category=motionCategory){
  const entries=motionCatalog.map(c=>({id:c.id,label:c.label,category:c.category,categoryLabel:c.category_label,description:c.source_description}));
  for(const [id,label] of Object.entries(MOTIONS))if(!isMocap(id))entries.push({id,label,category:'procedural',categoryLabel:'Procedurale · comparație'});
  entries.sort((a,b)=>MOTION_GROUP_ORDER.indexOf(a.category)-MOTION_GROUP_ORDER.indexOf(b.category));
  return category==='all'?entries:entries.filter(c=>c.category===category);
}
async function selectMotion(id,category=motionCategory){
  await operation(async()=>{
    await loadMocapClip(id);if(id!==motion)animationTime=0;
    motion=id;motionCategory=category;updateUI();
  },'Se încarcă mișcarea…');
}
function renderToolbar(){
  const bar=$('editToolbar');bar.innerHTML='';bar.hidden=!state||![1,5,6,7,8].includes(stage);
  if(!state)return;
  if(maskEditing&&editorKind!=='mask'){renderRegionEditor();return;}
  if(stage===1){
    if(!cropEdit){
      toolbarButton('Corectează dreptunghiul',()=>{cropEdit=true;cropDraft=[...state.meta.bbox];cropDirty=false;cropDrawMode=false;view='original';updateUI();});
      toolbarButton('Folosește imaginea întreagă',()=>operation(async()=>{const [w,h]=state.meta.source_size;await applyState(await api('/api/crop',{id:state.id,crop:[0,0,w,h]}));updateUI();},'Se actualizează decupajul…'));
    }else{
      toolbarButton('Trasează alt dreptunghi',()=>{cropDrawMode=true;renderToolbar();draw();},cropDrawMode);
      toolbarButton('Aplică decupajul',()=>operation(async()=>{
        if(!cropDraft)throw Error('Trasează mai întâi un dreptunghi.');
        await applyState(await api('/api/crop',{id:state.id,crop:cropDraft}));cropEdit=false;cropDirty=false;cropDraft=null;canvas.style.cursor='default';updateUI();
      },'Se aplică decupajul…'));
      toolbarButton('Anulează',()=>{cropEdit=false;cropDirty=false;cropDraft=null;canvas.style.cursor='default';updateUI();});
      const hint=document.createElement('span');hint.className='crop-hint';hint.textContent=cropDrawMode?'Trasează, apoi ajustează colțurile și laturile.':'Trage interiorul pentru mutare · colțurile și laturile pentru dimensiune · săgeți pentru ajustare fină';bar.append(hint);
    }
  }
  if(stage===5){
    if(!maskEditing){toolbarButton('Corectează masca cu pensula',()=>{editorKind='mask';initMaskEditor();maskEditing=true;view='overlay';refreshEditorPreview();updateUI();});return;}
    toolbarButton('+ Adaugă',()=>{brushMode='add';renderToolbar();},brushMode==='add');
    toolbarButton('− Șterge',()=>{brushMode='erase';renderToolbar();},brushMode==='erase');
    const l=document.createElement('label');l.textContent='Pensulă';const input=document.createElement('input');input.type='range';input.min=1;input.max=Math.max(2,Math.min(35,Math.round(Math.min(maskCanvas.width,maskCanvas.height)/3)));input.value=brushSize;input.setAttribute('aria-label','Dimensiunea pensulei');input.oninput=()=>brushSize=+input.value;l.append(input);bar.append(l);
    toolbarButton('Undo',()=>{const old=undo.pop();if(old){maskCanvas.getContext('2d').putImageData(old,0,0);maskDirty=true;refreshEditorPreview();draw();}});
    toolbarButton('Aplică masca',()=>operation(async()=>{await applyState(await api('/api/mask',{id:state.id,mask:maskCanvas.toDataURL('image/png')}));maskDirty=false;maskEditing=false;maskCanvas=null;view='overlay';updateUI();toast('Masca a fost aplicată. Reperele și rig-ul se vor recalcula.');},'Se actualizează masca…'));
    toolbarButton('Anulează',()=>{maskCanvas=null;maskDirty=false;maskEditing=false;view='overlay';updateUI();});
  }
  if(stage===6){
    const method=document.createElement('select');method.id='poseMethod';method.setAttribute('aria-label','Metoda de estimare a reperelor');
    const choices=[['auto','Automat · model / rezervă geometrică'],['neural',state.model?.profile==='finetuned'?'Model ajustat local · ResNet-50':'Model oficial · ResNet-50'],['shape','Din siluetă · comparație'],['proportions','Proporții · referință veche']];
    if(state.model?.candidate_available)choices.splice(2,0,['official','Model oficial · comparație'],['finetuned','Model ajustat · comparație']);
    if(state.example?.startsWith('garlic_'))choices.push(['reference','Garlic · adnotare de referință']);
    for(const [id,label] of choices){const o=document.createElement('option');o.value=id;o.textContent=label;o.selected=id===state.pose_method;o.disabled=id==='neural'&&!state.model?.available;method.append(o);}
    method.onchange=()=>operation(async()=>{await applyState(await api('/api/pose-estimate',{id:state.id,method:method.value}));view='crop';comparePose=false;updateUI();},'Se estimează articulațiile…');bar.append(method);
    const select=document.createElement('select');select.setAttribute('aria-label','Reper selectat');
    state.joint_names.forEach((n,i)=>{const opt=document.createElement('option');opt.value=i;opt.textContent=`${i+1}. ${n}`;opt.selected=i===selectedJoint;select.append(opt);});
    select.onchange=()=>{selectedJoint=+select.value;if(view.startsWith('heatmap_'))view=`heatmap_${selectedJoint}`;updateUI();};bar.append(select);
    if(state.meta.pose_comparison)checkbox('Compară cu geometria',comparePose,v=>{comparePose=v;view='crop';updateUI();});
    toolbarButton('Reinițializează reperele',()=>operation(async()=>{await applyState(await api('/api/pose-reset',{id:state.id}));updateUI();},'Se reinițializează reperele…'));
    toolbarButton('Editează corpul rigid',()=>startRegionEditor('body'));
    toolbarButton(bodyBoxEdit?'Trage dreptunghiul corpului':'Desenează corpul',()=>{bodyBoxEdit=!bodyBoxEdit;view='crop';renderViewButtons();renderToolbar();draw();},bodyBoxEdit);
    toolbarButton('Corp automat',()=>operation(async()=>{await applyState(await api('/api/body',{id:state.id,mask:null}));view='body_overlay';updateUI();},'Se reface corpul automat…'));
    const hint=document.createElement('span');hint.textContent='Trage un punct · săgeți pentru ajustare fină';hint.style.color='#778975';bar.append(hint);
  }
  if(stage===7){
    checkbox('Triunghiuri',showMesh,v=>{showMesh=v;draw();});checkbox('Schelet',showSkeleton,v=>{showSkeleton=v;draw();});
    toolbarButton('Corectează piesele',()=>startRegionEditor('parts'));
    toolbarButton('Piese automate',()=>operation(async()=>{await api('/api/parts',{id:state.id,labels:null});await applyState(await api('/api/step',{id:state.id,step:7}));updateUI();},'Se recalculează piesele…'));
  }
  if(stage===8){
    const playback=toolbarButton(playing?'Ⅱ Pauză':'▷ Redă',()=>{playing=!playing;renderToolbar();});playback.id='playPauseBtn';
    const categories=document.createElement('select');categories.id='motionCategory';categories.setAttribute('aria-label','Categoria animației');
    const all=motionChoices('all'),groups=new Map();
    for(const c of all){if(!groups.has(c.category))groups.set(c.category,{label:c.categoryLabel,count:0});groups.get(c.category).count++;}
    categories.add(new Option(`Toate · ${all.length}`, 'all',false,motionCategory==='all'));
    const groupOrder=MOTION_GROUP_ORDER;
    for(const id of groupOrder){const group=groups.get(id);if(group)categories.add(new Option(`${group.label} · ${group.count}`,id,false,motionCategory===id));}
    categories.onchange=()=>{const choices=motionChoices(categories.value);selectMotion(choices.some(c=>c.id===motion)?motion:choices[0].id,categories.value);};bar.append(categories);
    const choices=motionChoices(),select=document.createElement('select');select.id='motionSelect';select.setAttribute('aria-label','Mișcarea personajului');
    const optionGroups=new Map();
    if(motionCategory==='all')for(const id of groupOrder){const entry=groups.get(id);if(entry){const group=document.createElement('optgroup');group.label=entry.label;select.append(group);optionGroups.set(id,group);}}
    for(const c of choices){
      let parent=select;
      if(motionCategory==='all'){
        parent=optionGroups.get(c.category);
      }
      const o=new Option(c.label,c.id,false,c.id===motion);o.title=c.description||c.label;parent.append(o);
    }
    select.onchange=()=>selectMotion(select.value);bar.append(select);
    for(const [offset,label,symbol] of [[-1,'Animația anterioară','‹'],[1,'Animația următoare','›']]){
      const b=toolbarButton(symbol,()=>selectMotion(choices[(choices.findIndex(c=>c.id===motion)+offset+choices.length)%choices.length].id));b.setAttribute('aria-label',label);b.title=label;
    }
    if(isMocap(motion)){
      const projection=document.createElement('select');projection.id='motionProjection';projection.setAttribute('aria-label','Proiecția mișcării');
      for(const [id,label] of [['auto','Automat · planuri PCA'],['front','Frontal · X / Y'],['side','Lateral · Z / Y']]){const o=document.createElement('option');o.value=id;o.textContent=label;o.selected=id===motionProjection;projection.append(o);}
      projection.onchange=()=>{motionProjection=projection.value;updateUI();};bar.append(projection);
      checkbox('Schelet sursă CMU',showMotionSource,v=>{showMotionSource=v;updateUI();});
      const seek=document.createElement('label');seek.className='motion-seek';seek.textContent='Moment';
      const input=document.createElement('input');input.type='range';input.min=0;input.max=motionDuration(motion)-1/30;input.step=1/30;input.id='motionSeek';input.setAttribute('aria-label','Momentul înregistrării CMU');input.value=((animationTime%motionDuration(motion))+motionDuration(motion))%motionDuration(motion);
      input.oninput=()=>{playing=false;animationTime=+input.value;$('playPauseBtn').textContent='▷ Redă';draw();};seek.append(input);const time=document.createElement('span');time.id='motionTime';seek.append(time);bar.append(seek);
    }
    const mode=document.createElement('select');mode.id='renderMode';mode.setAttribute('aria-label','Modul de animație');
    for(const [id,label] of Object.entries(RENDER_MODES)){const o=document.createElement('option');o.value=id;o.textContent=label;o.selected=id===renderMode;mode.append(o);}
    mode.onchange=()=>{renderMode=mode.value;renderMetrics();renderToolbar();draw();};bar.append(mode);
    checkbox('Compară cu LBS',comparison,v=>{comparison=v;if(v&&renderMode==='baseline')renderMode='puppet';document.querySelector('.canvas-card').classList.toggle('comparing',v);renderMetrics();renderToolbar();draw();});
    checkbox('Schelet',showSkeleton,v=>{showSkeleton=v;draw();});
    if(renderMode==='puppet')checkbox('Membre în față',limbsFront,v=>{limbsFront=v;draw();});
    if(renderMode==='puppet')checkbox('Îmbinări netede',smoothJoints,v=>{smoothJoints=v;draw();});
    toolbarButton('↓ PNG',()=>exportCurrentFrame()).title='Descarcă postura curentă pe fundal transparent, fără schelet';
    toolbarButton('↓ PNG animat',()=>exportAnimation('apng')).title='Animație APNG transparentă · 2 secunde';
    toolbarButton('↓ Cadre PNG · ZIP',()=>exportAnimation('frames')).title='40 de cadre PNG transparente, numerotate, fără GIF';
    toolbarButton('↓ GIF',()=>exportGif()).title='Animație de 2 secunde pe fundal alb';
  }
}
function renderPartsLegend(){
  const legend=$('partLegend');legend.innerHTML='';legend.hidden=!state||stage<6||stage>8;
  if(legend.hidden)return;
  const title=document.createElement('h2');title.textContent='Zonele corpului';const note=document.createElement('p');note.textContent='Aceleași culori apar pe desen. Stânga și dreapta se referă la ecran.';
  const list=document.createElement('div');list.className='part-legend-grid';
  for(const p of state.parts){const item=document.createElement('span');item.className='part-legend-item';const dot=document.createElement('i');dot.style.background=`rgb(${p.color.join(',')})`;dot.style.color=p.color[0]*.299+p.color[1]*.587+p.color[2]*.114>145?'#143e32':'#fff';dot.textContent=String(p.id+1);const label=document.createElement('span');label.textContent=p.name;item.append(dot,label);list.append(item);}
  legend.append(title,note,list);
}
function startRegionEditor(kind){
  editorKind=kind;initMaskEditor(kind==='body'?'body_mask':'part_labels');maskEditing=true;
  view=kind==='body'?'body_overlay':'parts';refreshEditorPreview();renderViewButtons();renderToolbar();draw();
}
function renderRegionEditor(){
  const bar=$('editToolbar');
  if(editorKind==='parts'){
    const select=document.createElement('select');select.setAttribute('aria-label','Piesa de pictat');select.id='partPaint';
    state.parts.forEach(p=>{const o=document.createElement('option');o.value=p.id;o.textContent=p.name;o.selected=p.id===partPaint;select.append(o);});select.onchange=()=>partPaint=+select.value;bar.append(select);
  }else{
    toolbarButton('+ Protejează',()=>{brushMode='add';renderToolbar();},brushMode==='add');
    toolbarButton('− Elimină din corp',()=>{brushMode='erase';renderToolbar();},brushMode==='erase');
  }
  const l=document.createElement('label');l.textContent='Pensulă';const size=document.createElement('input');size.type='range';size.min=2;size.max=45;size.value=brushSize;size.oninput=()=>brushSize=+size.value;l.append(size);bar.append(l);
  toolbarButton('Undo',()=>{const old=undo.pop();if(old){maskCanvas.getContext('2d').putImageData(old,0,0);maskDirty=true;refreshEditorPreview();draw();}});
  toolbarButton(editorKind==='body'?'Aplică corpul':'Aplică piesele',()=>operation(async()=>{
    const body=editorKind==='body';const payload=body?{id:state.id,mask:maskCanvas.toDataURL()}:{id:state.id,labels:maskCanvas.toDataURL()};
    const result=await api(body?'/api/body':'/api/parts',payload);
    await applyState(body?result:await api('/api/step',{id:state.id,step:7}));
    maskEditing=false;maskDirty=false;maskCanvas=null;editorPreview=null;view=body?'body_overlay':'parts';updateUI();toast(body?'Corpul rigid a fost actualizat.':'Piesele au fost reconstruite.');
  },'Se aplică regiunea și se refac legăturile…'));
  toolbarButton('Anulează',()=>{maskEditing=false;maskDirty=false;maskCanvas=null;editorPreview=null;updateUI();});
}
function renderMetrics(){
  const m=state?.meta;const list=[];
  if(m){
    if(stage===0)list.push(['Pixeli de intrare',m.source_size.join(' × '),'după normalizare'],['Canale de culoare','RGB','8 biți / canal'],['Rezultat urmărit','2D','formă + structură + mișcare']);
    if(stage===1){const b=m.bbox;list.push(['Lățime decupaj',b[2]-b[0]+' px','limite pe axa x'],['Înălțime decupaj',b[3]-b[1]+' px','limite pe axa y'],['Metodă','Geometrică','contrast, fără rețea neuronală']);}
    if(stage===2)list.push(['Canal de intensitate','min RGB','aceeași regulă pentru toate imaginile'],['Grilă fundal','6 × 6','regulă fixă'],['Percentilă fundal','90','regulă fixă']);
    if(stage===3)list.push(['Prag calculat',m.threshold,'Otsu · estimat din histogramă'],['Niveluri posibile','256','intensitate 0…255'],['Alegere manuală','Nu','pentru acest prag']);
    if(stage===4)list.push(['Grosime estimată',m.stroke_width_px+' px','mediana secvențelor scurte'],['Nucleu ales',`${m.kernel} × ${m.kernel}`,'impar, limitat la 1…7'],['Operație','Închidere','dilatare → eroziune']);
    if(stage===5)list.push(['Arie selectată',(m.foreground_fraction*100).toFixed(1)+'%','din decupaj; nu este acuratețe'],['Metoda măștii',({alpha:'Alpha PNG',color:'Fundal RGB',otsu:'Otsu',manual:'Manuală'})[m.segmentation?.method]||'Otsu',m.segmentation?.note||'segmentare geometrică'],['Corecție manuală',m.mask_edited?'Da':'Nu','editează peste desen în Selecție']);
    if(stage===6){
      list.push(['Repere','15',state.pose_backend==='neural'?'din 17 heatmap-uri':'verifică punctele și traseele'],['Corp protejat',m.body_pixels+' px',m.pose_diagnostics?.body_manual?'selecție corectată':m.pose_diagnostics?.body_method||'partiționare geometrică'],['Estimare',state.pose_backend==='neural'?'ResNet-50':state.pose_backend==='reference'?'Adnotare':state.pose_backend==='proportions'?'Proporții':'Din siluetă',state.pose_backend==='neural'?`${m.neural_pose?.seconds??'—'} s · CPU`:'metodă geometrică']);
      if(m.pose_comparison)list.push(['Dezacord metode',m.pose_comparison.mean_difference_px+' px','distanță medie · nu acuratețe']);
    }
    if(stage>=7)list.push(['Vârfuri',m.vertices,'mesh pentru comparație'],['Piese rigide','9','corp + 8 segmente de membre'],['Mod selectat',renderMode==='puppet'?'Marionetă':renderMode==='protected'?'Corp rigid':'LBS clasic','schimbă modul la animație']);
    if(stage===8&&isMocap(motion)){const clip=getMocapClip(motion);list.push(['Captură reală',`${clip.subject} / ${clip.trial}`,'subiect / înregistrare CMU'],['Durată',clip.duration.toFixed(2)+' s',`${clip.source_fps} Hz sursă · ${clip.fps} Hz redare`],['Proiecție',motionProjection==='auto'?'PCA':motionProjection==='front'?'Frontală':'Laterală',motionProjection==='auto'?`brațe ${clip.planes.arms.angle_degrees}° · picioare ${clip.planes.legs.angle_degrees}°`:'direcții 3D proiectate în 2D']);}
  }
  $('metrics').innerHTML='';
  for(const [label,value,note] of list){const div=document.createElement('div');div.className='metric';const a=document.createElement('span'),b=document.createElement('strong'),c=document.createElement('small');a.textContent=label;b.textContent=value;c.textContent=note;div.append(a,b,c);$('metrics').append(div);}
  $('deformationSection').hidden=stage!==8;
  if(stage===8)updateDeformationReadout(true);
}
function drawHistogram(){
  const c=$('histogram'),ctx=c.getContext('2d'),hist=state?.meta.histogram;if(!hist)return;
  const w=c.clientWidth,h=210,ratio=Math.min(window.devicePixelRatio||1,2);if(w<60)return;
  c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const left=12,right=w-12,top=26,bottom=h-25,pw=right-left,ph=bottom-top;
  const bins=Array.from({length:64},(_,i)=>hist.slice(i*4,i*4+4).reduce((a,b)=>a+b,0));const max=Math.max(...bins.map(n=>Math.log1p(n)));
  ctx.strokeStyle='#dce4d5';ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=bottom-ph*i/3;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();}
  bins.forEach((n,i)=>{const bar=Math.log1p(n)/(max||1)*ph;ctx.fillStyle=i*4<=state.meta.threshold?'#478f79':'#c5d2bf';ctx.fillRect(left+i*pw/64,bottom-bar,Math.max(.8,pw/64-1),bar);});
  const x=left+state.meta.threshold/255*pw;ctx.strokeStyle='#096e57';ctx.lineWidth=2;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(x,top-10);ctx.lineTo(x,bottom);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#096e57';ctx.font='600 12px Segoe UI';ctx.fillText('T = '+state.meta.threshold,Math.max(left,Math.min(right-55,x+5)),14);
  ctx.fillStyle='#65785e';ctx.font='11px Segoe UI';ctx.textAlign='center';for(const value of [0,64,128,192,255])ctx.fillText(String(value),left+value/255*pw,bottom+18);ctx.textAlign='left';
  $('thresholdValue').textContent=state.meta.threshold;
}
function initMaskEditor(key='mask'){
  const im=currentImage(key);maskCanvas=document.createElement('canvas');maskCanvas.width=im.naturalWidth;maskCanvas.height=im.naturalHeight;maskCanvas.getContext('2d').drawImage(im,0,0);undo=[];maskDirty=false;
  const source=document.createElement('canvas');source.width=im.naturalWidth;source.height=im.naturalHeight;source.getContext('2d').drawImage(currentImage('mask'),0,0);editorOriginalMask=source.getContext('2d').getImageData(0,0,source.width,source.height).data;
  if(editorKind==='mask')brushSize=Math.max(1,Math.min(brushSize,Math.round(Math.min(im.naturalWidth,im.naturalHeight)*.035)));
}
function refreshEditorPreview(){
  if(!maskCanvas)return;
  editorPreview=document.createElement('canvas');editorPreview.width=maskCanvas.width;editorPreview.height=maskCanvas.height;
  const pixels=maskCanvas.getContext('2d').getImageData(0,0,maskCanvas.width,maskCanvas.height),out=editorPreview.getContext('2d').createImageData(maskCanvas.width,maskCanvas.height);
  for(let i=0;i<pixels.data.length;i+=4){
    const value=pixels.data[i],inside=editorOriginalMask[i]>127;
    if(editorKind==='parts'){
      const color=state.parts[Math.min(value,8)]?.color||[245,247,240];
      out.data[i]=inside?color[0]:245;out.data[i+1]=inside?color[1]:247;out.data[i+2]=inside?color[2]:240;out.data[i+3]=255;
    }else{out.data[i]=22;out.data[i+1]=160;out.data[i+2]=101;out.data[i+3]=(editorKind==='mask'||inside)?value*.42:0;}
  }
  editorPreview.getContext('2d').putImageData(out,0,0);
}

async function go(target){
  if(busy)return;if(maskDirty||cropDirty){toast('Aplică sau anulează editarea înainte de a continua.');return;}
  if(target<0||target>9||(!state&&target>0))return;
  if(state&&state.computed<target){
    await operation(async()=>{await applyState(await api('/api/step',{id:state.id,step:target}));stage=target;resetView();updateUI();},'Se calculează: '+STEPS[target].short.toLowerCase()+'…');
  }else{stage=target;resetView();updateUI();}
}
async function load(body){
  if(maskDirty||cropDirty){toast('Aplică sau anulează editarea înainte de a încărca alt desen.');return;}
  await operation(async()=>{await applyState(await api('/api/load',body));stage=0;animationTime=0;resetView();updateUI();},'Se încarcă desenul…');
}
function fitCamera(c,im,animation=false,exporting=false){
  const ratio=exporting?1:Math.min(window.devicePixelRatio||1,2);
  const width=exporting?c.width:c.clientWidth,height=exporting?c.height:c.clientHeight;
  if(width<60||height<60)return null;
  if(!exporting&&(c.width!==Math.round(width*ratio)||c.height!==Math.round(height*ratio))){c.width=Math.round(width*ratio);c.height=Math.round(height*ratio);}
  const bounds=animation&&state?.mesh?mocapBounds(state.mesh,motion,motionProjection,im.naturalWidth,im.naturalHeight):{x0:0,y0:0,x1:im.naturalWidth,y1:im.naturalHeight};
  const bw=bounds.x1-bounds.x0,bh=bounds.y1-bounds.y0;
  const s=Math.min((width-50)/bw,(height-40)/bh)*(animation&&!isMocap(motion) ? .79 : 1);
  return {ratio,width,height,s,ox:(width-bw*s)/2-bounds.x0*s,oy:(height-bh*s)/2-bounds.y0*s};
}
function draw(time=animationTime,targetCanvas=canvas,exporting=false,transparent=false){
  const ctx=targetCanvas.getContext('2d');
  if(!state){ctx.clearRect(0,0,targetCanvas.width,targetCanvas.height);return;}
  const im=currentImage(exporting?'texture':view);if(!im)return;
  const animated=exporting||stage===8;
  const cam=fitCamera(targetCanvas,im,animated,exporting);if(!cam)return;if(!exporting)camera=cam;
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,targetCanvas.width,targetCanvas.height);
  if(exporting&&!transparent){ctx.fillStyle='#ffffff';ctx.fillRect(0,0,targetCanvas.width,targetCanvas.height);}
  ctx.setTransform(cam.ratio,0,0,cam.ratio,0,0);ctx.translate(cam.ox,cam.oy);ctx.scale(cam.s,cam.s);
  let joints=state.joints;
  if(!exporting&&stage===8&&isMocap(motion)){
    const clip=getMocapClip(motion),now=((time%clip.duration)+clip.duration)%clip.duration;
    if($('motionTime'))$('motionTime').textContent=`${now.toFixed(2)} / ${clip.duration.toFixed(2)} s`;
    if($('motionSeek')&&document.activeElement!==$('motionSeek'))$('motionSeek').value=now;
    if(showMotionSource){drawMocapSource($('mocapSourceCanvas'),motion,time,state.parents);$('mocapSourceLink').href=clip.source_url;$('mocapInfo').textContent=`${clip.label} · subiect ${clip.subject}, înregistrarea ${clip.trial} · ASF + AMC · ${clip.source_frames} cadre la ${clip.source_fps} Hz · CMU: ${clip.source_description}`;}
  }
  if(animated&&comparison&&!exporting&&state.mesh){
    ctx.setTransform(cam.ratio,0,0,cam.ratio,0,0);
    const stacked=cam.width<600;
    const panels=stacked?[[0,0,cam.width,cam.height/2],[0,cam.height/2,cam.width,cam.height/2]]:[[0,0,cam.width/2,cam.height],[cam.width/2,0,cam.width/2,cam.height]];
    ["baseline",renderMode].forEach((mode,i)=>drawAnimationPanel(ctx,mode,time,...panels[i]));
    updateDeformationReadout();return;
  }
  if(animated&&state.mesh){
    joints=paintAnimation(ctx,renderMode,time);
  }else if(stage===5&&maskEditing&&maskCanvas){ctx.drawImage(currentImage('crop'),0,0);ctx.drawImage(editorPreview,0,0);}
  else if(maskEditing&&editorKind==='body'){ctx.drawImage(currentImage('crop'),0,0);ctx.drawImage(editorPreview,0,0);}
  else if(maskEditing&&editorKind==='parts'){ctx.drawImage(editorPreview,0,0);}
  else{ctx.drawImage(im,0,0);}
  if(exporting)return;
  if(stage===1&&view==='original'&&state.meta.bbox){
    const box=cropEdit&&cropDraft?cropDraft:state.meta.bbox;
    const [x0,y0,x1,y1]=box;ctx.strokeStyle='#10735e';ctx.lineWidth=2/cam.s;ctx.fillStyle='#10735e12';ctx.fillRect(x0,y0,x1-x0,y1-y0);ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    if(cropEdit){
      for(const [,x,y] of cropHandles(box)){ctx.fillStyle='white';ctx.fillRect(x-5/cam.s,y-5/cam.s,10/cam.s,10/cam.s);ctx.strokeRect(x-5/cam.s,y-5/cam.s,10/cam.s,10/cam.s);}
      ctx.font=`600 ${12/cam.s}px Segoe UI`;const text=`${Math.round(x1-x0)} × ${Math.round(y1-y0)} px`;ctx.fillStyle='#0d6853';ctx.fillText(text,x0,Math.max(14/cam.s,y0-10/cam.s));
    }
  }
  if(stage===7&&state.mesh&&showMesh&&view==='texture'&&!maskEditing){
    ctx.strokeStyle='#12705b80';ctx.lineWidth=.7/cam.s;
    for(const tri of state.mesh.triangles){ctx.beginPath();tri.forEach((i,k)=>{const [x,y]=state.mesh.vertices[i];if(k)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.closePath();ctx.stroke();}
  }
  if(stage===6&&drag?.type==='bodybox'){
    const a=drag.start,b=drag.last;ctx.fillStyle='#14946630';ctx.strokeStyle='#137458';ctx.lineWidth=2/cam.s;ctx.fillRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);
  }
  if(stage===6&&comparePose&&view==='crop'&&state.meta.pose_comparison&&!maskEditing){
    const geometric=state.meta.pose_comparison.geometric_joints;ctx.save();ctx.strokeStyle='#8b58a5';ctx.lineWidth=1.8/cam.s;ctx.setLineDash([5/cam.s,4/cam.s]);
    for(const [a,b] of state.bones){ctx.beginPath();ctx.moveTo(...geometric[a]);ctx.lineTo(...geometric[b]);ctx.stroke();}ctx.restore();
  }
  if(((stage===6&&(['crop','body_overlay'].includes(view)||view.startsWith('heatmap_'))&&!maskEditing)||((stage===7||stage===8)&&showSkeleton&&!maskEditing&&view!=='parts'))&&joints){
    ctx.lineWidth=2.2/cam.s;ctx.lineJoin='round';
    for(const [a,b] of state.bones){ctx.strokeStyle=b>=3&&b<=5||b>=9&&b<=11?'#db8051':'#16816b';ctx.beginPath();ctx.moveTo(...joints[a]);ctx.lineTo(...joints[b]);ctx.stroke();}
    joints.forEach(([x,y],i)=>{
      ctx.beginPath();ctx.arc(x,y,(stage===6&&i===selectedJoint?7:4.5)/cam.s,0,2*Math.PI);ctx.fillStyle=stage===6&&i===selectedJoint?'#f3b963':'#fff';ctx.fill();ctx.lineWidth=1.8/cam.s;ctx.strokeStyle=stage===6&&(state.meta.pose_diagnostics?.quality?.[i]??1)<.5?'#c6803d':'#176c58';ctx.stroke();
      if(stage===6){ctx.fillStyle='#123f33';ctx.font=`${10/cam.s}px Segoe UI`;ctx.fillText(String(i+1),x+7/cam.s,y-5/cam.s);}
    });
  }
  if(stage===8)updateDeformationReadout();
}
function paintAnimation(ctx,mode,time){
  const rig=state.mesh,target=animateJoints(rig,time,motion,amplitude,mode,motionProjection);
  if(mode==='puppet'){drawPuppet(ctx,rig,target,currentImage,limbsFront,isMocap(motion)?mocapDepthOrder(motion,time):null,smoothJoints);return target;}
  const positions=skinVertices(rig,target,mode),texture=currentImage(mode==='protected'?'limbs_texture':'texture');
  for(const tri of rig.triangles)affineTriangle(ctx,texture,tri.map(i=>rig.vertices[i]),tri.map(i=>positions[i]));
  if(mode==='protected')drawRigidPiece(ctx,currentImage('piece_0'),boneTransforms(rig,target)[0]);
  return target;
}
function drawAnimationPanel(ctx,mode,time,x,y,w,h){
  const im=currentImage('texture');ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();
  ctx.fillStyle=mode==='baseline'?'#8d624b':'#116f59';ctx.font='600 12px Segoe UI';ctx.fillText(mode==='baseline'?'LBS CLASIC':mode==='puppet'?'MARIONETĂ RIGIDĂ':'CORP RIGID',x+15,y+24);
  const bounds=mocapBounds(state.mesh,motion,motionProjection,im.naturalWidth,im.naturalHeight),bw=bounds.x1-bounds.x0,bh=bounds.y1-bounds.y0;
  const scale=Math.min((w-32)/bw,(h-62)/bh)*(isMocap(motion)?1:.88);
  ctx.translate(x+(w-bw*scale)/2-bounds.x0*scale,y+43+(h-57-bh*scale)/2-bounds.y0*scale);ctx.scale(scale,scale);
  const joints=paintAnimation(ctx,mode,time);
  if(showSkeleton){ctx.lineWidth=1.5/scale;ctx.strokeStyle='#14806f';for(const [a,b] of state.bones){ctx.beginPath();ctx.moveTo(...joints[a]);ctx.lineTo(...joints[b]);ctx.stroke();}}
  ctx.restore();
}
function updateDeformationReadout(force=false){
  if(!state?.mesh||stage!==8)return;if(!force&&performance.now()-lastMetric<180)return;lastMetric=performance.now();
  const rig=state.mesh,baseline=measureDeformation(rig,animateJoints(rig,animationTime,motion,amplitude,'baseline',motionProjection),'baseline');
  const selected=measureDeformation(rig,animateJoints(rig,animationTime,motion,amplitude,renderMode,motionProjection),renderMode);
  $('baselineError').textContent=baseline.bodyEdgesChecked?baseline.bodyMaxPercent.toFixed(2)+'%':'—';
  $('selectedError').textContent=selected.bodyEdgesChecked?selected.bodyMaxPercent.toFixed(2)+'%':'—';
  $('selectedModeLabel').textContent=RENDER_MODES[renderMode];
  $('limbError').textContent='Variație maximă pe segmentele de membre: '+selected.limbMaxPercent.toFixed(2)+'%.';
}
function pointerLocation(e){
  const r=canvas.getBoundingClientRect();if(!camera)return [0,0];
  return [(e.clientX-r.left-camera.ox)/camera.s,(e.clientY-r.top-camera.oy)/camera.s];
}
function clampPoint(p){const im=currentImage('crop');return [Math.max(0,Math.min(im.naturalWidth-1,p[0])),Math.max(0,Math.min(im.naturalHeight-1,p[1]))];}
function paint(from,to){
  if(editorKind==='parts'){
    const ctx=maskCanvas.getContext('2d'),pixels=ctx.getImageData(0,0,maskCanvas.width,maskCanvas.height);
    const x0=Math.max(0,Math.floor(Math.min(from[0],to[0])-brushSize)),x1=Math.min(maskCanvas.width-1,Math.ceil(Math.max(from[0],to[0])+brushSize));
    const y0=Math.max(0,Math.floor(Math.min(from[1],to[1])-brushSize)),y1=Math.min(maskCanvas.height-1,Math.ceil(Math.max(from[1],to[1])+brushSize));
    const dx=to[0]-from[0],dy=to[1]-from[1],den=dx*dx+dy*dy;
    for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
      const t=den?Math.max(0,Math.min(1,((x-from[0])*dx+(y-from[1])*dy)/den)):0;
      const d=Math.hypot(x-from[0]-t*dx,y-from[1]-t*dy),index=(y*maskCanvas.width+x)*4;
      if(d<=brushSize/2&&editorOriginalMask[index]>127){pixels.data[index]=partPaint;pixels.data[index+1]=partPaint;pixels.data[index+2]=partPaint;pixels.data[index+3]=255;}
    }
    ctx.putImageData(pixels,0,0);maskDirty=true;refreshEditorPreview();draw();return;
  }
  const ctx=maskCanvas.getContext('2d');ctx.strokeStyle=brushMode==='add'?'white':'black';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=brushSize;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...to);ctx.stroke();ctx.beginPath();ctx.arc(...to,brushSize/2,0,Math.PI*2);ctx.fill();maskDirty=true;draw();
  if(editorKind==='body'||editorKind==='mask'){refreshEditorPreview();draw();}
}
canvas.addEventListener('pointerdown',e=>{
  if(busy||!state)return;const p=pointerLocation(e);canvas.focus();
  if(stage===1&&cropEdit){
    const [w,h]=state.meta.source_size,handle=cropDrawMode?'draw':hitCrop(cropDraft,p,12/camera.s);
    if(handle){drag={type:'crop',handle,start:p,last:p,base:[...cropDraft]};if(handle==='draw')cropDraft=cropRect(p,p,w,h);}
  }
  else if(stage===6&&bodyBoxEdit){drag={type:'bodybox',start:p,last:p};}
  else if(maskEditing){if(undo.length>=15)undo.shift();undo.push(maskCanvas.getContext('2d').getImageData(0,0,maskCanvas.width,maskCanvas.height));drag={type:'paint',last:p};paint(p,p);}
  else if(stage===6&&['crop','body_overlay'].includes(view)){
    let nearest=-1,best=18/camera.s;state.joints.forEach((q,i)=>{const d=Math.hypot(q[0]-p[0],q[1]-p[1]);if(d<best){best=d;nearest=i;}});
    if(nearest>=0){selectedJoint=nearest;drag={type:'joint'};renderToolbar();draw();}
  }
  if(drag)canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove',e=>{
  if(busy)return;const p=pointerLocation(e);
  if(!drag){if(stage===1&&cropEdit){const hit=cropDrawMode?'draw':hitCrop(cropDraft,p,12/camera.s);canvas.style.cursor=hit==='draw'?'crosshair':hit==='move'?'grab':hit?hit+'-resize':'default';}return;}
  if(drag.type==='crop'){
    const [w,h]=state.meta.source_size;drag.last=p;
    cropDraft=drag.handle==='draw'?cropRect(drag.start,p,w,h):drag.handle==='move'?moveCrop(drag.base,p[0]-drag.start[0],p[1]-drag.start[1],w,h):resizeCrop(drag.base,drag.handle,p,w,h);
    cropDirty=true;
  }
  if(drag.type==='bodybox')drag.last=p;
  if(drag.type==='paint'){paint(drag.last,p);drag.last=p;}
  if(drag.type==='joint')state.joints[selectedJoint]=clampPoint(p);
  draw();
});
async function endPointer(){
  if(!drag)return;const previous=drag;drag=null;
  if(previous.type==='crop'){
    if(cropDraft[2]-cropDraft[0]<8||cropDraft[3]-cropDraft[1]<8){cropDraft=previous.base;toast('Dreptunghiul trebuie să aibă cel puțin 8 × 8 pixeli.');}
    cropDrawMode=false;renderToolbar();draw();
  }
  if(previous.type==='joint')await persistPose();
  if(previous.type==='bodybox'){
    bodyBoxEdit=false;
    const c=document.createElement('canvas'),im=currentImage('crop');c.width=im.naturalWidth;c.height=im.naturalHeight;
    const ctx=c.getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='white';
    const a=previous.start,b=previous.last;ctx.fillRect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]));
    await operation(async()=>{await applyState(await api('/api/body',{id:state.id,mask:c.toDataURL()}));view='body_overlay';updateUI();},'Se protejează corpul selectat…');
  }
}
canvas.addEventListener('pointerup',endPointer);canvas.addEventListener('pointercancel',endPointer);
async function persistPose(){await operation(async()=>{await applyState(await api('/api/pose',{id:state.id,joints:state.joints}));updateUI();},'Se actualizează reperul…');}

function downloadBlob(blob,name){const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),3000);}
function exportUrl(kind){if(state){operation(async()=>{
  if(maskDirty||cropDirty)throw Error('Aplică sau anulează editarea înainte de salvare.');
  await api('/api/settings',{id:state.id,settings:{render_mode:renderMode,motion,motion_projection:motionProjection,show_motion_source:showMotionSource,comparison,limbs_front:limbsFront,smooth_joints:smoothJoints}});
  const a=document.createElement('a');a.href=`/api/export/${state.id}/${kind}`;a.download='';a.click();
},'Se pregătește proiectul pentru salvare…');}}
async function exportCurrentFrame(){
  if(!state?.mesh)return;
  await operation(async()=>{
    if(maskDirty||cropDirty)throw Error('Aplică sau anulează editarea înainte de export.');
    const c=document.createElement('canvas');c.width=600;c.height=600;
    draw(animationTime,c,true,true);
    const blob=await new Promise((resolve,reject)=>c.toBlob(value=>value?resolve(value):reject(Error('Cadrul PNG nu a putut fi salvat.')),'image/png'));
    downloadBlob(blob,`ACI-cadru-${renderMode}.png`);toast('Cadru PNG exportat · fundal transparent.');
  },'Se pregătește cadrul PNG transparent…');
}
async function exportGif(){return exportAnimation('gif');}
async function exportAnimation(kind){
  if(!state?.mesh)return;
  const options={gif:{label:'GIF-ul',extension:'gif',transparent:false},apng:{label:'PNG-ul animat',extension:'png',transparent:true},frames:{label:'cadrele PNG',extension:'zip',transparent:true}}[kind];
  if(!options)return;
  await operation(async()=>{
    if(maskDirty||cropDirty)throw Error('Aplică sau anulează editarea înainte de export.');
    if(kind!=='gif'){
      const health=await(await fetch('/api/health')).json();
      if(!health.animation_exports?.includes(kind))throw Error('Salvează proiectul JSON și repornește aplicația pentru noile exporturi PNG.');
    }
    const c=document.createElement('canvas');c.width=600;c.height=600;const frames=[];
    const start=isMocap(motion)?animationTime:0;
    for(let i=0;i<40;i++){draw(start+i/20,c,true,options.transparent);frames.push(c.toDataURL('image/png'));if(i%4===0){$('busyText').textContent=`Se pregătesc ${options.label}: ${i+1}/40 cadre…`;await new Promise(r=>setTimeout(r,0));}}
    $('busyText').textContent=`Se salvează ${options.label}…`;
    const response=await fetch(`/api/${kind}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames})});
    if(!response.ok){const e=await response.json().catch(()=>({}));throw Error(e.error||'Exportul nu a putut fi salvat.');}
    const name=kind==='frames'?'cadre':'animatie';
    downloadBlob(await response.blob(),`ACI-${name}-${renderMode}.${options.extension}`);
    toast(kind==='frames'?'40 de cadre PNG transparente exportate în ZIP.':`${kind==='apng'?'PNG animat transparent':'GIF'} exportat · ${RENDER_MODES[renderMode]} · 2 secunde.`);
  },'Se pregătește animația…');
}

$('prevBtn').onclick=()=>go(stage-1);$('nextBtn').onclick=()=>go(stage+1);
$('runAllBtn').onclick=()=>{
  if(maskDirty||cropDirty){toast('Aplică mai întâi editarea curentă.');return;}
  operation(async()=>{
    for(let i=state.computed+1;i<=8;i++){
      $('busyText').textContent=`Pasul ${i+1}/9 · ${STEPS[i].short}…`;
      await applyState(await api('/api/step',{id:state.id,step:i}));
    }
    stage=8;resetView();updateUI();toast('Flux complet calculat. Poți reveni la orice etapă.');
  },'Se calculează etapele…');
};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;renderLesson();});
function presentation(){document.body.classList.toggle('presentation');$('presentationBtn').setAttribute('aria-pressed',String(document.body.classList.contains('presentation')));setTimeout(draw,50);}
$('presentationBtn').onclick=presentation;
$('aboutContent').innerHTML=METHOD_HTML;$('aboutBtn').onclick=()=>$('aboutDialog').showModal();$('closeDialog').onclick=()=>$('aboutDialog').close();
$('aboutDialog').onclick=e=>{if(e.target===$('aboutDialog')){$('aboutDialog').close();}};
for(const id of ['uploadBtn','emptyUploadBtn'])$(id).onclick=()=>$('fileInput').click();
async function fileData(file){if(file.size>12_000_000)throw Error('Imaginea trebuie să aibă sub 12 MB.');return new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(file);});}
$('fileInput').onchange=async()=>{const f=$('fileInput').files[0];if(!f)return;try{await load({image:await fileData(f),name:f.name});}catch(e){toast(e.message,true);}$('fileInput').value='';};
const area=$('canvasArea');area.addEventListener('dragover',e=>e.preventDefault());area.addEventListener('drop',async e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(!f||busy)return;try{await load({image:await fileData(f),name:f.name});}catch(err){toast(err.message,true);}});
$('saveBtn').onclick=()=>exportUrl('project');$('bundleBtn').onclick=()=>exportUrl('zip');$('rigBtn').onclick=()=>exportUrl('rig');$('gifBtn').onclick=()=>exportGif();
$('framePngBtn').onclick=()=>exportCurrentFrame();$('apngBtn').onclick=()=>exportAnimation('apng');$('framesBtn').onclick=()=>exportAnimation('frames');
$('importBtn').onclick=()=>$('projectInput').click();$('projectInput').onchange=async()=>{
  const f=$('projectInput').files[0];if(!f)return;
  await operation(async()=>{if(f.size>18_000_000)throw Error('Proiectul JSON este prea mare.');const p=JSON.parse(await f.text());await applyState(await api('/api/import',p));stage=Math.min(state.computed,8);resetView();updateUI();toast('Proiect reîncărcat cu masca și reperele salvate.');},'Se importă proiectul…');$('projectInput').value='';
};
$('downloadImageBtn').onclick=()=>{
  if([8,9].includes(stage)&&state?.mesh){exportCurrentFrame();}
  else if([1,6,7].includes(stage)){canvas.toBlob(blob=>downloadBlob(blob,`ACI-pas-${stage+1}.png`));}
  else{const im=currentImage(view);if(!im)return;const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;c.getContext('2d').drawImage(im,0,0);c.toBlob(blob=>downloadBlob(blob,`ACI-${view}.png`));}
};
window.addEventListener('keydown',async e=>{
  if($('aboutDialog').open||busy||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;
  if(stage===1&&cropEdit&&e.target===canvas&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
    e.preventDefault();const d=e.shiftKey?10:1,[w,h]=state.meta.source_size;cropDraft=moveCrop(cropDraft,e.key==='ArrowLeft'?-d:e.key==='ArrowRight'?d:0,e.key==='ArrowUp'?-d:e.key==='ArrowDown'?d:0,w,h);cropDirty=true;draw();return;
  }
  if(stage===6&&!maskEditing&&!bodyBoxEdit&&e.target===canvas&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
    e.preventDefault();const p=[...state.joints[selectedJoint]],amount=e.shiftKey?10:1;
    if(e.key==='ArrowLeft')p[0]-=amount;if(e.key==='ArrowRight')p[0]+=amount;if(e.key==='ArrowUp')p[1]-=amount;if(e.key==='ArrowDown')p[1]+=amount;
    state.joints[selectedJoint]=clampPoint(p);draw();await persistPose();return;
  }
  if(e.key==='ArrowRight'){e.preventDefault();go(stage+1);}if(e.key==='ArrowLeft'){e.preventDefault();go(stage-1);}
  if(e.code==='Space'&&stage===8){e.preventDefault();playing=!playing;renderToolbar();}
  if(e.key.toLowerCase()==='f'&&!e.ctrlKey&&!e.metaKey)presentation();
});
new ResizeObserver(()=>{draw();if(stage===3)drawHistogram();}).observe($('canvasArea'));
function tick(now){const dt=Math.min((now-previousTick)/1000,.1);previousTick=now;if(stage===8&&playing&&!busy){animationTime+=dt;draw();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
window.labSnapshot=()=>({stage,computed:state?.computed,busy,cropEdit,cropDirty,cropDraft:cropDraft?[...cropDraft]:null,maskDirty,editorKind,maskEditing,renderMode,comparison,comparePose,motion,motionCategory,motionProjection,showMotionSource,playing,poseMethod:state?.pose_method,poseBackend:state?.pose_backend,model:state?.model,view,animationTime,joints:state?.joints,meta:state?.meta,mesh:state?.mesh,id:state?.id});

async function init(){
  updateUI();
  try{
    const health=await(await fetch('/api/health')).json();
    if(health.version!=='3.8')throw Error('Este pornit serverul vechi. Închide fereastra serverului și pornește din nou START_PROIECT.cmd pentru versiunea 3.8.');
    const motionResponse=await fetch('/api/motions');if(!motionResponse.ok)throw Error('Catalogul CMU nu este disponibil.');
    motionCatalog=await motionResponse.json();for(const clip of motionCatalog)MOTIONS[clip.id]=clip.label;
    const response=await fetch('/api/examples');if(!response.ok)throw Error('Server indisponibil.');const examples=await response.json();
    for(const ex of examples){const b=document.createElement('button');b.className='example';b.dataset.example=ex.id;b.title=ex.note;const im=document.createElement('img');im.src=`/samples/${ex.id}.png`;im.alt=ex.name;const title=document.createElement('span');title.textContent=ex.name.replace('Garlic · ','');b.append(im,title);b.onclick=()=>load({example:ex.id});$('examples').append(b);}
    await load({example:'garlic_roi'});
  }catch(e){toast('Pornește aplicația prin START_PROIECT.cmd. '+e.message,true);}
}
init();
