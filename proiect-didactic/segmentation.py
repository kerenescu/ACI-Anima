"""Mască din alpha sau fundal RGB uniform; fără model neuronal de segmentare.

Otsu rămâne comparația pentru fotografii pe hârtie. Nicio valoare nu este
cerută utilizatorului: zgomotul și culoarea fundalului se estimează la margine.
"""
import numpy as np


def color_background(image):
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    border=np.concatenate([rgb[0],rgb[-1],rgb[1:-1,0],rgb[1:-1,-1]])
    color=np.median(border,axis=0)
    noise=np.max(np.abs(border-color),axis=1)
    median=float(np.median(noise));mad=float(np.median(np.abs(noise-median)))
    q80=float(np.percentile(noise,80))
    threshold=float(max(2,np.ceil(q80+3*mad)))
    uniform=bool(q80<=10 and threshold<=25)
    distance=np.max(np.abs(rgb-color),axis=2)
    foreground=distance>threshold
    return foreground,dict(background_rgb=color.round(1).tolist(),background_noise_q80=round(q80,2),
                           color_threshold=threshold,uniform=uniform)


def mask_from_alpha(alpha):
    a=np.asarray(alpha.convert('L'))
    # Otsu pe opacitate elimină urmele aproape transparente și oscilațiile
    # Lanczos, care altfel pot lega mai multe personaje printr-un singur pixel.
    from pipeline import otsu
    threshold,_,_=otsu(a)
    return a>threshold,threshold


def choose_mask(image,legacy,alpha=None):
    if alpha is not None:
        mask,threshold=mask_from_alpha(alpha)
        return mask,dict(method='alpha',description='Transparența PNG originală',alpha_threshold=threshold,
            note='Prag automat pe opacitate; nu pe culoarea hainelor. Otsu pe intensitate rămâne comparația.')
    candidate,stats=color_background(image)
    if stats['uniform'] and candidate.any() and candidate.mean()<.9:
        # Aceleași funcții geometrice ca în pipeline, importate la apel pentru
        # a evita dependența circulară la încărcarea modulelor.
        from pipeline import fill_holes,components
        mask,count=components(fill_holes(candidate))
        return mask,dict(method='color',description='Contrast RGB față de fundalul uniform',
            note='Păstrăm culorile deschise care diferă de fundal. Otsu rămâne comparația.',
            components=count,**stats)
    return legacy.copy(),dict(method='otsu',description='Otsu + închidere + umplere',
        note='Fundalul nu este suficient de uniform pentru separare RGB; folosim metoda pentru hârtie.',**stats)
