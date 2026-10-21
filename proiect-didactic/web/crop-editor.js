export function cropRect(a,b,width,height){
 const x=v=>Math.max(0,Math.min(width,v)),y=v=>Math.max(0,Math.min(height,v));
 return [x(Math.min(a[0],b[0])),y(Math.min(a[1],b[1])),x(Math.max(a[0],b[0])),y(Math.max(a[1],b[1]))];
}
export function cropHandles([x0,y0,x1,y1]){
 const xm=(x0+x1)/2,ym=(y0+y1)/2;
 return [['nw',x0,y0],['n',xm,y0],['ne',x1,y0],['e',x1,ym],['se',x1,y1],['s',xm,y1],['sw',x0,y1],['w',x0,ym]];
}
export function hitCrop(rect,[x,y],radius){
 for(const [id,hx,hy] of cropHandles(rect))if(Math.hypot(x-hx,y-hy)<=radius)return id;
 return x>=rect[0]&&x<=rect[2]&&y>=rect[1]&&y<=rect[3]?'move':null;
}
export function moveCrop(rect,dx,dy,width,height){
 dx=Math.max(-rect[0],Math.min(width-rect[2],dx));dy=Math.max(-rect[1],Math.min(height-rect[3],dy));
 return [rect[0]+dx,rect[1]+dy,rect[2]+dx,rect[3]+dy];
}
export function resizeCrop(rect,handle,[x,y],width,height){
 const out=[...rect];
 if(handle.includes('w'))out[0]=Math.max(0,Math.min(out[2]-8,x));
 if(handle.includes('e'))out[2]=Math.min(width,Math.max(out[0]+8,x));
 if(handle.includes('n'))out[1]=Math.max(0,Math.min(out[3]-8,y));
 if(handle.includes('s'))out[3]=Math.min(height,Math.max(out[1]+8,y));
 return out;
}
