# ACI-Anima

Aplicație locală în română: încarci un desen, corectezi masca și articulațiile, construiești rigul și animezi personajul. Include explicații tehnice pentru fiecare etapă și 35 de animații CMU.

## Instalare și pornire

Sunt necesare Python 3.10–3.12 și un browser modern.

```powershell
git clone https://github.com/kerenescu/ACI-Anima.git
cd ACI-Anima/proiect-didactic
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python server.py --open
```

Pentru recunoașterea neuronală a articulațiilor, rulează **INSTALARE_MODEL.cmd** o singură dată, cu acces la internet. Descarcă modelul oficial și instalează PyTorch CPU. Ulterior pornești prin **START_PROIECT.cmd**; procesarea și animațiile funcționează offline. Fără model, aplicația indică folosirea metodei geometrice. Greutățile și mediul Python se instalează local și sunt excluse din Git.

În aplicație: **Rulează toți pașii → Animează → alege mișcarea**. Poți reveni la fiecare etapă pentru corecții și explicații. Salvează proiectul JSON înainte de oprirea serverului; acesta păstrează desenul și corecțiile. GIF-ul exportat poate fi introdus în PowerPoint.

În modul Marionetă, **Îmbinări netede** acoperă întreaga secțiune a tăieturii cu racorduri Bézier și capace rotunjite, dimensionate după grosimea membrului și îndoire. Construcția rămâne fără încrucișări inclusiv la îndoiri inverse. Dacă desenul are contur întunecat, estimăm culoarea și grosimea lui și îl continuăm pe marginea vizibilă a volumului. Fragmentele conturului vechi ajunse în interior sunt acoperite local, iar un filtru Gaussian estompează contactul piesă–racord. Fiecare pixel original aparține unei singure piese, pentru a evita capetele duplicate la rotire. Opțiunea este activă implicit, se păstrează în proiect și se aplică și exportului GIF.

## Surse

Demonstrator inspirat de [Smith et al., *A Method for Animating Children’s Drawings of the Human Figure*, 2023](https://doi.org/10.1145/3592788). Modelul și Garlic provin din [Meta AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings), cu [licența MIT păstrată](proiect-didactic/assets/LICENSE_AnimatedDrawings.txt). Exemplele `fig6_*` sunt extrase din figura 6 a articolului pentru explicația metodei.

Mișcările provin din [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/), susținută prin NSF Grant #0196217. Fișierul `assets/motions.json` conține datele 3D de redare și identificatorii/URL-urile surselor. Aplicația folosește segmentare alpha/RGB/Otsu, predictor ResNet-50 și marionetă/LBS; diferențele față de articol sunt explicate în interfață.
