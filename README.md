# TalTech Active Recall Agent

Viitepohine oppimisagent kursusematerjalide jaoks. Rakendus loeb lokaalsed materjalid sisse, ehitab neist otsitava teadmistebaasi ja kuvab vastused koos allikaviidetega.

## Mida see teeb

- loeb failid `data/` kaustast sisse
- tukeldab materjalid vaiksemateks loikudeks
- leiab loikudest peamised moisted ehk konseptsioonid
- ehitab kohaliku otsinguindeksi
- kuvab Streamliti vaates viited, chunk-id ja lehekuljed

Fookus on viidetel, mitte vabalt genereeritud kokkuvotetel.

## Failid

- `app.py`  
  Streamliti kasutajaliides. Kuvab reziimid, otsib teadmistebaasist vasteid ja naitab viiteid.

- `knowledge_base.py`  
  Loeb materjalid sisse, tukeldab teksti, tuvastab konseptsioonid, ehitab otsinguindeksi ja kirjutab `generated/` valjundid.

- `index_materials.py`  
  Eraldi skript teadmistebaasi uuendamiseks. Sobib kasitsi kaivitamiseks voi Task Scheduleri alla.

- `query_synonyms.json`  
  Paringulaiendid. Siia saab lisada kursusepohiseid synonym'e.

- `data/`  
  Sisendmaterjalid. Toetatud on `.pdf`, `.txt` ja `.md`.

- `generated/`  
  Automaatselt loodud vahefailid:
  - `catalog.json`
  - `overview.md`
  - `sources/*.md`
  - `concepts/*.md`

## Kuidas see tootab

1. `index_materials.py` kutsub `build_knowledge_base(...)`.
2. `knowledge_base.py` loeb `data/` kausta failid sisse.
3. Tekst jagatakse chunk'ideks.
4. Iga chunki jaoks leitakse sagedasemad sisulised terminid.
5. Chunk'idest ehitatakse TF-IDF otsinguindeks.
6. `app.py` kasutab seda indeksit, et kuvada teemale sobivad allikad ja viited.

## Kaivitamine

Koik kasud eeldavad, et oled enne projektikausta liikunud:

```powershell
cd C:\taltech-agent
```

### 1. Loo virtuaalkeskkond

```powershell
py -3 -m venv .venv
```

### 2. Aktiveeri virtuaalkeskkond

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Paigalda soltuvused

```powershell
py -3 -m pip install -r requirements.txt
```

### 4. Pane materjalid `data/` kausta

Naiteks:
- loenguslaidid PDF-ina
- markdown-markmed
- tekstifailid voi transkriptsioonid

### 5. Ehita teadmistebaas

```powershell
py -3 .\index_materials.py
```

### 6. Kaivita rakendus

```powershell
py -3 -m streamlit run .\app.py
```

## Ollama tugi

Ollama on valikuline. Seda kasutatakse ainult osades vaadetes ja ainult siis, kui lokaalne Ollama server tootab.

Vaikimisi eeldab rakendus:
- `OLLAMA_EXE=%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe`
- mudel `qwen2.5:3b`

Kui tahad mudelit ette alla tombata:

```powershell
& "$env:USERPROFILE\AppData\Local\Programs\Ollama\ollama.exe" pull qwen2.5:3b
```

Kui Ollamat pole, tootab rakendus edasi viitepohise variandina.

## Updating

Teadmistebaasi ei pea igal kaivitusel uuesti ehitama. Tavakasutuses piisab sellest, et `generated/` on juba olemas.

Tee rebuild siis, kui:
- lisad `data/` kausta uusi faile
- muudad olemasolevaid materjale `data/` kaustas
- muudad `query_synonyms.json` faili
- muudad `knowledge_base.py` loogikat
- `generated/` kaust puudub voi on kustutatud

Rebuildi ei ole vaja teha siis, kui:
- avad lihtsalt appi uuesti
- esitad uusi kusimusi samade materjalide pohjal
- muudad ainult `app.py` kasutajaliidest

Teadmistebaasi uuendamine:

```powershell
cd C:\taltech-agent
py -3 .\index_materials.py
```

## Mida tiim peaks teadma

See on MVP, mitte loplik tootetoode.

Tugevused:
- tootab lokaalselt
- ei vaja OpenAI API votit
- naitab allikaviiteid koos lehekulgedega
- on laiendatav erinevate oppimisreziimidega

Piirangud:
- konseptsioonide ekstraktsioon on heuristiline
- otsing ei ole veel semantiline
- transcriptide timecode tugi puudub
- testid puuduvad

## Jagamisel oluline

- `.venv/`, `.idea/`, `__pycache__/` ja `generated/` ei pea repoga kaasa minema
- `generated/` ehitatakse uuesti käsuga `py -3 .\index_materials.py`
- allikad identifitseeritakse nüüd `data/` suhtelise tee järgi, mitte ainult failinime järgi

Kui `py` mingil pohjusel puudub, kasuta otse virtuaalkeskkonna Pythonit:

```powershell
.\.venv\Scripts\python.exe .\index_materials.py
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```
