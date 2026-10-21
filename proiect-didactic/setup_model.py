"""Pregătește numai predictorul oficial, fără dataset și fără servicii cloud."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import sys
import urllib.request
from neural_pose import sha256_file

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')

ROOT=Path(__file__).resolve().parent
URL='https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar'


def cleanup_downloads(models):
    for name in ['drawn_humanoid_pose_estimator.mar','best_AP_epoch_72.pth']:
        (models/name).unlink(missing_ok=True)


def prepare():
    import torch
    import numpy as np
    models=ROOT/'models';models.mkdir(exist_ok=True)
    weights=models/'pose_resnet50.pt';manifest=models/'model.json'
    if weights.exists() and manifest.exists():
        digest=sha256_file(weights)
        if digest==json.loads(manifest.read_text(encoding='utf-8'))['sha256']:
            cleanup_downloads(models)
            print('Modelul este deja pregătit și verificat.');return
    archive=models/'drawn_humanoid_pose_estimator.mar'
    if not archive.exists():
        print('Se descarcă modelul oficial (~375 MB). Datasetul nu este descărcat.',flush=True)
        request=urllib.request.Request(URL,headers={'User-Agent':'ACI-Drawing-Lab/3.0'})
        temporary=archive.with_suffix('.part')
        with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as target:
            while chunk:=response.read(1024*1024):target.write(chunk)
        temporary.replace(archive)
    with ZipFile(archive) as z:
        # Extragem numai checkpointul necesar, fără configurații sau handlers.
        for name in ['best_AP_epoch_72.pth']:
            target=models/name
            if not target.exists():
                with z.open(name) as source,target.open('wb') as dest:
                    while chunk:=source.read(1024*1024):dest.write(chunk)
    # Checkpoint vechi cu un scalar NumPy în metadate. Permitem numai aceste
    # tipuri numerice; păstrăm weights_only=True, fără execuția configurației.
    scalar=np._core.multiarray.scalar if hasattr(np,'_core') else np.core.multiarray.scalar
    with torch.serialization.safe_globals([(scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype('float64'))]):
        checkpoint=torch.load(models/'best_AP_epoch_72.pth',map_location='cpu',weights_only=True,mmap=True)
    temporary=weights.with_suffix('.tmp')
    torch.save(checkpoint['state_dict'],temporary)
    temporary.replace(weights)
    digest=sha256_file(weights)
    manifest.write_text(json.dumps(dict(name='AnimatedDrawings ResNet-50',sha256=digest,source=URL,
        license='MIT',input_size=[192,256],heatmap_size=[48,64],channels=17),indent=2),encoding='utf-8')
    cleanup_downloads(models)
    print('Greutăți pregătite (~136 MB), cu verificare SHA-256. Repornește aplicația.')


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
