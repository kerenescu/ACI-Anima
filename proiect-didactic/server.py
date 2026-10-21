"""Server local: python server.py --open. Nu necesită Flask sau internet."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid
import webbrowser

import numpy as np
from PIL import Image
from pipeline import (ROOT, EXAMPLES, DrawingSession, decode_image, normalize,
                      png_bytes, restore_project, set_crop, set_pose,set_body,set_parts,set_settings)
from mocap import catalog,get_clip
from neural_pose import model_status

SESSIONS: dict[str, DrawingSession] = {}
LOCK = threading.RLock()
STATIC = ROOT / 'web'
MAX_BODY = 40_000_000


def add_session(s):
    with LOCK:
        # Sesiunile sunt volatile; exportul JSON păstrează proiectul pe disc.
        if len(SESSIONS) >= 24:
            oldest = min(SESSIONS, key=lambda k: SESSIONS[k].created)
            del SESSIONS[oldest]
        sid = uuid.uuid4().hex; SESSIONS[sid] = s; return sid


class Handler(BaseHTTPRequestHandler):
    server_version = 'ACIDrawingLab/3.8'

    def log_message(self, format, *args):
        if self.path.startswith('/api/') and '/image/' not in self.path:
            super().log_message(format, *args)

    def send(self, code, data, mime='application/json; charset=utf-8', filename=None):
        if isinstance(data, (dict, list)): data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        if isinstance(data, str): data = data.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if filename: self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers(); self.wfile.write(data)

    def session(self, sid):
        if sid not in SESSIONS: raise ValueError('Sesiunea a expirat. Încarcă desenul sau importă proiectul salvat.')
        return SESSIONS[sid]

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/api/health':
                return self.send(200, {'ok': True, 'version': '3.8','model':model_status(),'mocap_clips':len(catalog())})
            if path == '/api/motions':return self.send(200,catalog())
            if path.startswith('/api/motion/'):
                motion=path.rsplit('/',1)[-1]
                if motion not in {c['id'] for c in catalog()}:return self.send(404,{'error':'Mișcare CMU necunoscută.'})
                return self.send(200,get_clip(motion))
            if path == '/api/examples': return self.send(200, EXAMPLES)
            if path.startswith('/api/image/'):
                _, _, _, sid, key = path.split('/')
                with LOCK:
                    s = self.session(sid)
                    if key not in s.images: raise ValueError('Imaginea nu a fost calculată încă.')
                    return self.send(200, png_bytes(s.images[key]), 'image/png')
            if path.startswith('/api/export/'):
                parts = path.split('/'); sid, kind = parts[3:5]
                with LOCK:
                    s = self.session(sid)
                    if kind == 'zip': return self.send(200, s.bundle(), 'application/zip', 'ACI-rezultate.zip')
                    if kind == 'project': return self.send(200, s.project(), filename='ACI-proiect.json')
                    if kind == 'rig': return self.send(200, s.mesh or {}, filename='ACI-rig.json')
                raise ValueError('Tip de export necunoscut.')
            if path.startswith('/samples/'):
                name = path.rsplit('/', 1)[-1]
                if name not in {x['id']+'.png' for x in EXAMPLES}: return self.send(404, {'error':'Exemplu inexistent.'})
                return self.send(200, (ROOT/'assets'/'samples'/name).read_bytes(), 'image/png')
            allowed = {'/':('index.html','text/html; charset=utf-8'),
                       '/app.js':('app.js','text/javascript; charset=utf-8'),
                       '/motion.js':('motion.js','text/javascript; charset=utf-8'),
                       '/mocap.js':('mocap.js','text/javascript; charset=utf-8'),
                       '/crop-editor.js':('crop-editor.js','text/javascript; charset=utf-8'),
                       '/content.js':('content.js','text/javascript; charset=utf-8'),
                       '/style.css':('style.css','text/css; charset=utf-8')}
            if path == '/favicon.ico': return self.send(204, b'', 'image/x-icon')
            if path in allowed:
                filename,mime=allowed[path];return self.send(200,(STATIC/filename).read_bytes(),mime)
            return self.send(404, {'error':'Resursă inexistentă.'})
        except (ValueError,KeyError) as e:
            return self.send(400, {'error':str(e)})
        except (BrokenPipeError,ConnectionResetError): pass

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
                return self.send(403, {'error':'Cererea trebuie trimisă din aplicația locală.'})
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_BODY: return self.send(413, {'error':'Cerere prea mare sau fără conținut.'})
            body=json.loads(self.rfile.read(size));path=urlsplit(self.path).path
            with LOCK:
                if path=='/api/load':
                    example=body.get('example')
                    if example:
                        item=next((x for x in EXAMPLES if x['id']==example),None)
                        if not item:raise ValueError('Exemplu invalid.')
                        image=Image.open(ROOT/'assets'/'samples'/(example+'.png'))
                        name=item['name']
                    else:
                        image=decode_image(body.get('image'));name=str(body.get('name','Desen propriu'))[:100]
                    s=DrawingSession(normalize(image),name,example);sid=add_session(s)
                elif path=='/api/import':
                    s=restore_project(body);sid=add_session(s)
                elif path=='/api/gif':
                    frames=body.get('frames',[])
                    if not 2<=len(frames)<=48:raise ValueError('Exportul necesită între 2 și 48 de cadre.')
                    decoded=[]
                    for frame in frames:
                        im=decode_image(frame,1_500_000).convert('RGB')
                        if max(im.size)>800:raise ValueError('Cadrele GIF sunt limitate la 800 px.')
                        decoded.append(im)
                    if len({im.size for im in decoded})!=1:raise ValueError('Cadrele trebuie să aibă aceeași dimensiune.')
                    out=BytesIO()
                    decoded[0].save(out,format='GIF',save_all=True,append_images=decoded[1:],duration=50,
                                    loop=0,optimize=False,disposal=2)
                    return self.send(200,out.getvalue(),'image/gif','ACI-animatie.gif')
                else:
                    sid=str(body.get('id',''));s=self.session(sid)
                    if path=='/api/step':s.run(int(body.get('step',0)))
                    elif path=='/api/crop':
                        set_crop(s,body.get('crop'));s.run(1)
                    elif path=='/api/mask':
                        s.run(5);mask=np.asarray(decode_image(body.get('mask')).convert('L'))>127
                        s.set_mask(mask,True);s.invalidate(6)
                    elif path=='/api/pose':
                        s.run(6);set_pose(s,body.get('joints'))
                    elif path=='/api/pose-reset':
                        s.invalidate(6);s.run(6)
                    elif path=='/api/pose-estimate':
                        method=body.get('method','auto');s.run(5)
                        s.estimate_pose(method);s.computed=6;s.revision+=1
                    elif path=='/api/body':
                        value=body.get('mask')
                        set_body(s,np.asarray(decode_image(value).convert('L'))>127 if value else None)
                    elif path=='/api/parts':
                        value=body.get('labels');set_parts(s,np.asarray(decode_image(value).convert('L')) if value else None)
                    elif path=='/api/settings':set_settings(s,body.get('settings',{}))
                    else:return self.send(404,{'error':'Acțiune inexistentă.'})
            return self.send(200,s.state(sid))
        except (ValueError,TypeError,KeyError,OverflowError,OSError) as e:
            return self.send(400,{'error':str(e)})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            import traceback;traceback.print_exc()
            return self.send(500,{'error':'Operația nu a reușit. Detaliile apar în fereastra serverului.'})


def main():
    parser=argparse.ArgumentParser(description='ACI Drawing Lab · aplicație didactică locală')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--open',action='store_true',help='Deschide browserul după pornirea serverului')
    args=parser.parse_args()
    server=None
    for port in range(args.port,args.port+12):
        try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler);break
        except OSError:continue
    if server is None:raise SystemExit('Nu există un port liber. Încearcă --port 9000.')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'\nACI Drawing Lab\n{url}\nDoar pe acest calculator. Ctrl+C pentru oprire.\n',flush=True)
    if args.open:threading.Timer(.6,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
