"""Datele de redare pentru cele 35 de capturi CMU incluse offline."""
from functools import lru_cache
import json
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent / 'assets/motions.json'

@lru_cache(maxsize=1)
def library():
    data = json.loads(LIBRARY.read_text(encoding='utf-8'))
    if {item['id'] for item in data['catalog']} != set(data['clips']):
        raise ValueError('Catalogul și animațiile CMU nu corespund.')
    return data

def catalog():
    return library()['catalog']

def get_clip(motion):
    try:
        return library()['clips'][motion]
    except KeyError:
        raise ValueError('Mișcare CMU necunoscută.') from None
