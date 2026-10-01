# U-Bahn plan matcher

Prototype that maps scanned Berlin U-Bahn construction plans to a U-Bahn station with a vision-language
model (HTW Berlin API, OpenAI-compatible). Plans are processed only through that API and are not part of
this repository.

## Pipeline
1. `preprocess.py` finds text regions (title block, bottom right corner, further text blocks) and cuts them
   out at full resolution; the model also gets a reduced overview of the sheet.
2. `test_prototype.py` holds the prompt (six steps), the station list handling and the API call.
3. `postcheck.py` checks the answer in code: date rule, spelling against the station list, line check.
4. `webapp.py` is a local web app (folder batch, results table, ground-truth evaluation, reports in `reports/`).
5. `evaluation.py` computes accuracy, precision, recall, F1, coverage and per-confidence / per-station results.

`stationcollector.py` scrapes the station table `bhfdaten.csv` (source: berliner-u-bahn.info, see the notice in
the script). `test_htw_api.py`, `test_htw_prompt.py` and `test_htw_vision.py` are the first connection tests.

## Setup
```
pip install -r requirements.txt
export HTW_API_KEY="..."     # see .env.example
python webapp.py             # http://127.0.0.1:5050
```
## Workflow
1. Choose a folder with plans and start the analysis; the app shows the progress with a preview of the detected text regions.
2. Plans the model is unsure about (medium/low confidence, no station, corrected by a check, skipped) are listed for manual
   review: full-resolution zoomable plan, title block and text regions, then *use proposed station*, *own name* or *no station*.
3. *Create folders* copies every plan into a folder named after its station (`sorted_plans/<date>/<station>/`, plus `_overview.csv`).
   A plan that clearly belongs to several stations is copied into each of them; plans without a station go to
   `_no_station/<type of facility>/` (e.g. rectifier substation). Every page of a multi-page PDF is analyzed and sorted on its own.
   Several plans can be analyzed in parallel (Settings).
   Originals are never changed. The app keeps work copies of uploaded plans in the system temp folder (cleaned after 3 days).

Ground-truth files `groundtruth*.csv` (columns: Station name, Image name) in this folder appear in the app.
