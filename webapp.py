"""Lokale Web-Oberfläche für test_prototype.py.

Benötigt: pip install flask pillow pymupdf openai opencv-python-headless numpy
Start:    python webapp.py   ->   http://127.0.0.1:5050
"""
import csv
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from openai import APITimeoutError
from flask import Flask, Response, jsonify, send_file, render_template_string, request

import evaluation
import postcheck
import preprocess
import test_prototype as proto

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # Baupläne können sehr groß sein (TIFFs >100 MB)


@app.errorhandler(Exception)
def _json_errors(exc):
    """Fehler immer als JSON zurückgeben, damit die Oberfläche eine lesbare Meldung zeigt."""
    from werkzeug.exceptions import HTTPException

    if isinstance(exc, HTTPException):
        return jsonify(error=f"{exc.code} {exc.name}: {exc.description}"), exc.code
    return jsonify(error=f"{type(exc).__name__}: {exc}"), 500

try:
    from PIL import Image as _PILImage

    # Plan scans can exceed Pillow's default limit (~179 MP, guards against decompression
    # bombs). The files here are the user's own, so the check is switched off.
    _PILImage.MAX_IMAGE_PIXELS = None
except ImportError:
    pass

MIME = {"PNG": "image/png", "JPEG": "image/jpeg"}
WEB_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _encode(img, max_px=None, fmt="PNG", info=None):
    from PIL import Image

    # Bilevel scans (mode "1") become grayscale: 1 byte/pixel instead of 3 for RGB
    if img.mode == "1":
        img = img.convert("L")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_px:
        original = img.size
        # reducing_gap makes the downscale of huge scans much faster
        img.thumbnail((max_px, max_px), Image.LANCZOS, reducing_gap=2.0)
        if info is not None and img.size != original:
            info["scaled"] = {"from": list(original), "to": list(img.size)}
    out = io.BytesIO()
    if fmt == "JPEG":
        img.save(out, format="JPEG", quality=90)
    else:
        img.save(out, format="PNG")
    return out.getvalue()


def _to_web_image(data, filename, max_px=None, fmt="PNG", info=None, page=1):
    """Gibt (bytes, mime) zurück; PDF (gewählte Seite) und TIFF werden zu PNG konvertiert."""
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    if mime in WEB_MIME_TYPES and not max_px:
        return data, mime
    from PIL import Image  # nur für Formate wie TIF nötig

    if mime == "application/pdf":
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            raise RuntimeError("PDF support needs PyMuPDF: pip install pymupdf") from None

        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            idx = min(max(int(page or 1), 1), pdf.page_count) - 1
            img = Image.open(io.BytesIO(pdf[idx].get_pixmap(dpi=150).tobytes("png")))
            return _encode(img, max_px, fmt, info), MIME[fmt]
    return _encode(Image.open(io.BytesIO(data)), max_px, fmt, info), MIME[fmt]


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/netmap")
def netmap():
    return send_file(Path(__file__).parent / "ubahnlanes.png", mimetype="image/png")


@app.post("/preview")
def preview():
    """Vorschaubild für Formate, die der Browser nicht selbst anzeigt (TIFF, PDF)."""
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="No file."), 400
    try:
        png, _ = _to_web_image(file.read(), file.filename, max_px=2400, page=_page())
    except Exception as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500
    return Response(png, mimetype="image/png")


def _field(answer, label):
    m = re.search(r"^[\s*\-]*\*\*" + label + r":\*\*\s*(.*)$", answer or "", re.M | re.I)
    return m.group(1).replace("*", "").strip() if m else ""


def _is_none_but_maybe_station(answer):
    """True, wenn die Antwort "none" lautet, das Modell den Plan aber nicht als eigenständige Anlage einstuft."""
    station = _field(answer, "Station").lower()
    belongs = _field(answer, "Belongs to a station").lower()
    return bool(re.match(r"(none|n/a|not\b|unknown|unclear|no station)", station)) and not belongs.startswith("no")


_prep_cache = OrderedDict()  # (SHA-1 der Datei [+ "-pN"]) -> Prepared; die Oberfläche ruft /preprocess vor /analyze auf
_prep_lock = threading.Lock()


def _page():
    """Seite eines PDFs aus dem Formular (ab 1) oder None."""
    try:
        return max(1, int(request.form.get("page") or 0)) or None
    except ValueError:
        return None


def _prepared(data, filename, write_debug=True, page=None):
    """Preprocessing mit kleinem Cache (Schlüssel: Inhalt der Datei, bei PDF-Seiten mit Seitenzahl);
    das Debug-Overlay wird nur auf Wunsch geschrieben."""
    digest = hashlib.sha1(data).hexdigest()
    suffix = f"-p{page}" if page else ""
    key = digest + suffix
    with _prep_lock:
        if key in _prep_cache:
            _prep_cache.move_to_end(key)
            return _prep_cache[key]
    prep = preprocess.prepare(data, filename, page or 1)
    prep.plan_id = _store_upload(data, filename, digest) + suffix
    if write_debug:
        DEBUG_DIR.mkdir(exist_ok=True)  # Overlay mit den gefundenen Regionen zum Prüfen
        stem = re.sub(r"[^\w.\-]+", "_", filename) + (f"_p{page}" if page else "")
        (DEBUG_DIR / (stem + ".jpg")).write_bytes(prep.overlay)
    with _prep_lock:
        _prep_cache[key] = prep
        while len(_prep_cache) > 8:
            _prep_cache.popitem(last=False)
    return prep


@app.post("/preprocess")
def preprocess_preview():
    """Liefert das Debug-Overlay (Plan mit eingezeichneten Textregionen) für die Vorschau während der Analyse."""
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="No file."), 400
    try:
        prep = _prepared(file.read(), file.filename, page=_page())
    except Exception as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500
    resp = Response(prep.overlay, mimetype="image/jpeg")
    resp.headers["X-Plan-Id"] = prep.plan_id
    return resp


class _PreprocessOff(Exception):
    """Marker: Preprocessing wurde in der Oberfläche abgeschaltet."""


@app.post("/analyze")
def analyze():
    # Zeit, die die Oberfläche für das Preprocessing schon verbraucht hat, zählt zum Limit pro Plan
    t0 = time.monotonic() - float(request.form.get("time_used") or 0)
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="Please select an image."), 400

    stations_text = None
    if request.form.get("use_list") == "1":
        if not proto.STATIONS_CSV.exists():
            return jsonify(error=f"Station list not found: {proto.STATIONS_CSV.name}"), 500
        stations_text, _ = proto.load_stations_text(proto.STATIONS_CSV)

    try:
        data = file.read()
        page = _page()
        info = {"plan_id": _store_upload(data, file.filename) + (f"-p{page}" if page else "")}
        try:
            if request.form.get("preprocess") == "0":
                raise _PreprocessOff()
            prep = _prepared(data, file.filename, page=page)  # Textregionen in voller Auflösung (gecacht)
            images = prep.images
            info["preprocess"] = prep.info
        except Exception as exc:  # abgeschaltet oder fehlgeschlagen: ganzes Blatt wie bisher
            image, mime = _to_web_image(data, file.filename, max_px=proto.MAX_IMAGE_PX, fmt="JPEG", info=info, page=page)
            images = [{"caption": "Image 1: the whole sheet.", "data": image, "mime": mime}]
            info["preprocess"] = ({"disabled": True} if isinstance(exc, _PreprocessOff)
                                  else {"error": f"{type(exc).__name__}: {exc}"})
        thinking = request.form.get("thinking") == "1"
        limit = float(request.form.get("time_limit") or 0)  # Sekunden pro Plan, 0 = kein Limit
        left = lambda: max(0.5, limit - (time.monotonic() - t0)) if limit else None
        skipped = jsonify(skipped=True, reason=f"skipped: took longer than {limit:g} s", **info)
        if limit and time.monotonic() - t0 > limit - 2:  # schon das Preprocessing hat das Limit fast aufgebraucht
            return skipped
        try:
            result = proto.analyze_images(images, stations_text, thinking=thinking, timeout=left())
        except APITimeoutError:
            return skipped
        retried = second_look_timeout = False
        if request.form.get("retry") == "1" and not thinking and _is_none_but_maybe_station(result):
            remaining = left()
            if remaining is None or remaining > 5:
                try:
                    result = proto.analyze_images(images, stations_text, thinking=True, timeout=remaining)  # second look
                    retried = True
                except APITimeoutError:
                    second_look_timeout = True  # erste Antwort behalten
        corrections = []
        if stations_text:  # Prüfungen gegen die Stationsliste (Schreibweise, Linie)
            result, corrections = postcheck.PostCheck(proto.STATIONS_CSV).apply(result)
    except Exception as exc:  # Anzeige im UI statt Stacktrace
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500
    return jsonify(result=result, retried=retried, second_look_timeout=second_look_timeout,
                   corrections=corrections, **info)


REPORT_DIR = Path(__file__).parent / "reports"
DEBUG_DIR = Path(__file__).parent / "debug"  # Overlays der Textregionen-Erkennung
DEFAULT_GT = Path(__file__).parent / "groundtruth.csv"  # used when nothing else is chosen in the UI


def _gt_files():
    """Ground-Truth-Dateien im Projektordner (groundtruth*.csv), zur Auswahl in der Oberfläche."""
    return sorted(p.name for p in Path(__file__).parent.glob("*.csv") if p.name.lower().startswith("groundtruth"))
_report_lock = threading.Lock()


def _next_report_id():
    """Nächste fortlaufende Berichts-ID (höchste vorhandene ID im Ordner + 1)."""
    ids = [int(m.group(1)) for f in REPORT_DIR.glob("report_*.json") if (m := re.match(r"report_(\d+)", f.name))]
    return max(ids, default=0) + 1


@app.post("/report")
def report():
    """Speichert nach jedem Durchlauf einen Bericht (ID, Ergebnisse, bei Ground Truth auch Kennzahlen)."""
    rows = json.loads(request.form.get("rows", "[]"))
    meta = json.loads(request.form.get("meta", "{}"))

    # Auswahl in der Oberfläche: hochgeladene Datei > "none" (keine Auswertung) > Datei aus dem Projektordner > Standard
    choice = request.form.get("gt_choice", "")
    gt = request.files.get("ground_truth")
    gt_name, gt_data = (gt.filename, gt.read()) if gt else (None, None)
    if gt_data is None and choice != "none":
        path = Path(__file__).parent / choice if choice in _gt_files() else DEFAULT_GT  # nur bekannte Dateinamen
        if path.exists():
            gt_name, gt_data = path.name, path.read_bytes()

    rep = {"n": len(rows), "metrics": None}
    if gt_data:
        try:
            truth = evaluation.load_ground_truth(gt_data)
            rep = evaluation.evaluate(rows, truth, evaluation.StationNames(proto.STATIONS_CSV))
            rep["metrics"] = True
            meta["ground_truth"], meta["ground_truth_rows"] = gt_name, len(truth)
        except Exception as exc:
            rep["ground_truth_error"] = f"Ground truth could not be read: {exc}"
    rep["results"] = [{"document": r["title"], "station": r.get("station") or None,
                       "confidence": r.get("level") or None, "error": r.get("error") or None,
                       "downscaled": r.get("scaled"), "second_look": bool(r.get("retried")), "preprocess": r.get("prep"),
                       "corrections": r.get("corr") or [],
                       "answer": r.get("answer") or None} for r in rows]

    now = datetime.now()
    with _report_lock:  # IDs dürfen bei gleichzeitigen Läufen nicht doppelt vergeben werden
        REPORT_DIR.mkdir(exist_ok=True)
        rid = _next_report_id()
        stem = f"report_{rid:04d}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        rep["meta"] = {"id": rid, "timestamp": now.isoformat(timespec="seconds"), "model": proto.MODEL,
                       "max_image_px": proto.MAX_IMAGE_PX, **meta}
        rep["saved_as"] = f"reports/{stem}.json"
        (REPORT_DIR / f"{stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        judged = {i["document"]: i for i in rep.get("items", [])}
        with open(REPORT_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["document", "prediction", "confidence", "ground_truth", "correct", "downscaled", "second_look", "error"])
            for r in rep["results"]:
                j = judged.get(r["document"])
                sc = r["downscaled"]
                w.writerow([r["document"], r["station"] or "none", r["confidence"] or "",
                            (j["truth"] or "none") if j else "", j["correct"] if j else "",
                            f'{"x".join(map(str, sc["from"]))} -> {"x".join(map(str, sc["to"]))}' if sc else "",
                            r["second_look"], r["error"] or ""])
    return jsonify(rep)


# --- Manuelle Prüfung: Originalpläne, Textregionen, Sortierung in Ordner ------------------------------
# Hochgeladene Pläne werden im Temp-Ordner abgelegt, damit die Oberfläche sie nach dem Durchlauf in voller
# Auflösung zeigen und am Ende in Stationsordner sortieren kann (der Browser kennt keine Dateipfade).
WORK_DIR = Path(tempfile.gettempdir()) / "bip26_stationmatcher"
PLAN_DIR, VIEW_DIR = WORK_DIR / "plans", WORK_DIR / "views"
STATE_FILE = WORK_DIR / "state.json"
DEFAULT_EXPORT_DIR = Path(__file__).parent / "sorted_plans"
_gray_cache = OrderedDict()  # plan_id -> Graustufenbild in voller Auflösung (LRU, 2 Pläne)
_gray_lock = threading.Lock()


def _init_work_dir(max_age_days=3):
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_days * 86400
    for d in (PLAN_DIR, VIEW_DIR):  # alte Arbeitskopien aufräumen
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)


_init_work_dir()


def _store_upload(data, filename, digest=None):
    """Legt den hochgeladenen Plan ab und gibt seine plan_id (SHA-1-Anfang) zurück."""
    pid = (digest or hashlib.sha1(data).hexdigest())[:20]
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAN_DIR / (pid + Path(filename).suffix.lower())
    if not path.exists():
        path.write_bytes(data)
    return pid


def _split_pid(pid):
    """plan_id = SHA-1-Anfang der Datei, bei PDF-Seiten mit Zusatz "-pN" -> (Datei-ID, Seite oder None)."""
    m = re.fullmatch(r"([0-9a-f]{20})(?:-p(\d{1,4}))?", pid or "")
    if not m:
        raise FileNotFoundError("invalid plan id")
    return m.group(1), int(m.group(2)) if m.group(2) else None


def _plan_path(pid):
    base, _ = _split_pid(pid)
    found = sorted(PLAN_DIR.glob(base + ".*"))
    if not found:
        raise FileNotFoundError("plan not found (the work copy may have been cleaned up)")
    return found[0]


def _gray(pid):
    with _gray_lock:
        if pid in _gray_cache:
            _gray_cache.move_to_end(pid)
            return _gray_cache[pid]
    path = _plan_path(pid)
    img = preprocess.load_gray(path.read_bytes(), path.name, _split_pid(pid)[1] or 1)
    with _gray_lock:
        _gray_cache[pid] = img
        while len(_gray_cache) > 2:
            _gray_cache.popitem(last=False)
    return img


@app.get("/plan_image/<pid>")
def plan_image(pid):
    """Der ganze Plan in voller Auflösung (Graustufen-JPEG) für den Zoom-Betrachter."""
    cached = VIEW_DIR / f"{pid}.jpg"
    if not cached.exists():
        _gray(pid).save(cached, format="JPEG", quality=88)
    return send_file(cached, mimetype="image/jpeg", max_age=3600)


@app.get("/plan_regions/<pid>")
def plan_regions(pid):
    """Größe des Bildes und die erkannten Textregionen (Schriftfeld, Ecke, Textblöcke)."""
    base, page = _split_pid(pid)
    suffix = f"-p{page}" if page else ""
    with _prep_lock:
        prep = next((v for k, v in _prep_cache.items() if k.startswith(base) and k.endswith(suffix)
                     and (suffix or "-p" not in k)), None)
    if prep is None:  # Cache verdrängt oder App neu gestartet: neu berechnen (ohne Debug-Overlay zu schreiben)
        path = _plan_path(pid)
        prep = _prepared(path.read_bytes(), path.name, write_debug=False, page=page)
    return jsonify(size=list(prep.size), regions=prep.regions)


@app.get("/plan_crop/<pid>")
def plan_crop(pid):
    """Ausschnitt in Originalauflösung (PNG); x0, y0, x1, y1 in Pixeln von /plan_image."""
    img = _gray(pid)
    x0, y0, x1, y1 = (int(request.args.get(k, 0)) for k in ("x0", "y0", "x1", "y1"))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)
    if x1 - x0 < 1 or y1 - y0 < 1 or (x1 - x0) * (y1 - y0) > 80e6:
        return jsonify(error="invalid crop"), 400
    out = io.BytesIO()
    img.crop((x0, y0, x1, y1)).save(out, format="PNG")
    return Response(out.getvalue(), mimetype="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/state")
def get_state():
    if STATE_FILE.exists():
        return Response(STATE_FILE.read_text(encoding="utf-8"), mimetype="application/json")
    return jsonify(rows=[])


@app.post("/state")
def save_state():
    """Merkt den Stand des Durchlaufs (Ergebnisse, Entscheidungen), damit ein Neuladen der Seite nichts verliert."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(request.get_json(force=True), ensure_ascii=False), encoding="utf-8")
    return jsonify(ok=True)


def _station_names():
    """Anzeigenamen der heutigen Stationen (für die Eingabehilfe bei eigenen Namen)."""
    if not proto.STATIONS_CSV.exists():
        return []
    names = evaluation.StationNames(proto.STATIONS_CSV)
    with open(proto.STATIONS_CSV, newline="", encoding="utf-8") as f:
        shown = {names.canonical(r["name"])[1] for r in csv.DictReader(f) if not r.get("heute") and r.get("name")}
    return sorted(n for n in shown if n)


_last_export = None


@app.post("/reveal")
def reveal_export():
    """Öffnet den zuletzt erzeugten Zielordner im Dateimanager (nur dieser Ordner, nur lokal)."""
    path = Path((request.get_json(force=True) or {}).get("path", ""))
    if _last_export is None or path != _last_export or not path.is_dir():
        return jsonify(error="Unknown folder."), 400
    opener = ["open"] if sys.platform == "darwin" else ["xdg-open"] if sys.platform != "win32" else None
    if opener:
        subprocess.Popen(opener + [str(path)])
    elif hasattr(os, "startfile"):
        os.startfile(str(path))  # noqa: Windows
    return jsonify(ok=True)


def _folder_name(station):
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", station or "").strip(" .")
    return name[:80] or "_unassigned"


def _facility_folder(facility):
    """Ordnername für Pläne ohne Station: "_no_station/<Art der Anlage>"."""
    text = re.sub(r"\s*\(.*?\)", "", facility or "").split("–")[0].split(" - ")[0].strip(" .")
    text = text if text and text.lower() not in ("n/a", "none", "na") else "unclear location"
    return "_no_station/" + _folder_name(text[:1].upper() + text[1:])


@app.post("/pdf_pages")
def pdf_pages():
    """Seitenzahl eines PDFs (jede Seite wird als eigener Plan analysiert)."""
    file = request.files.get("image")
    if not file:
        return jsonify(error="No file."), 400
    import pymupdf

    with pymupdf.open(stream=file.read(), filetype="pdf") as pdf:
        return jsonify(pages=pdf.page_count)


def _copy_plan(src, pid, target):
    """Kopiert einen Plan; bei einer Seite eines mehrseitigen PDFs nur diese Seite als eigenes PDF."""
    _, page = _split_pid(pid)
    if page and src.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(src) as pdf:
            if pdf.page_count > 1:
                with pymupdf.open() as out:
                    out.insert_pdf(pdf, from_page=page - 1, to_page=page - 1)
                    out.save(target)
                return
    shutil.copy2(src, target)


@app.post("/export")
def export_folders():
    """Kopiert jeden Plan in den Ordner seiner Station(en) bzw. nach "_no_station/<Art>" und schreibt eine Übersicht."""
    body = request.get_json(force=True)
    out_dir = Path(os.path.expanduser(body.get("out_dir") or str(DEFAULT_EXPORT_DIR)))
    if not out_dir.is_absolute():
        return jsonify(error="Please enter an absolute folder path."), 400
    names = evaluation.StationNames(proto.STATIONS_CSV) if proto.STATIONS_CSV.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    copied, missing, folders, overview = 0, [], set(), []
    for item in body.get("items", []):
        stations = item.get("stations")
        if stations is None:  # ältere Form: ein Feld "station"
            stations = [item.get("station")] if item.get("station") else []
        targets = []
        for st in stations:
            st = (st or "").strip()
            if st and names:
                _, display = names.canonical(st)  # "Alexanderplatz [Bstg.A]" -> "Alexanderplatz", frühere Namen -> heute
                st = display or st
            if st and _folder_name(st) not in targets:
                targets.append(_folder_name(st))
        if not targets:
            targets = [_facility_folder(item.get("facility"))]
        try:
            src = _plan_path(item.get("plan_id"))
        except FileNotFoundError:
            missing.append(item.get("name"))
            continue
        name = Path(item.get("name") or src.name).name
        m = re.fullmatch(r"(.*)\.pdf \(page (\d+)\)", name, re.I)
        if m:  # "plan.pdf (page 2)" -> "plan (page 2).pdf"
            name = f"{m.group(1)} (page {m.group(2)}).pdf"
        for folder in targets:
            target_dir = out_dir / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            n = 2
            while target.exists():  # gleichnamige Pläne aus verschiedenen Unterordnern nicht überschreiben
                target = target_dir / f"{Path(name).stem} ({n}){Path(name).suffix}"
                n += 1
            _copy_plan(src, item.get("plan_id"), target)
            copied += 1
            folders.add(folder)
            overview.append([item.get("name"), folder, item.get("source", ""), item.get("confidence", "")])
    global _last_export
    _last_export = out_dir
    with open(out_dir / "_overview.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["document", "folder", "decided_by", "model_confidence"])
        w.writerows(overview)
    return jsonify(out_dir=str(out_dir), files=copied, folders=len(folders), missing=missing)


@app.get("/")
def index():
    return render_template_string(PAGE, model=proto.MODEL, gt_files=_gt_files(),
                                  default_gt=DEFAULT_GT.name if DEFAULT_GT.exists() else "",
                                  station_names=_station_names(),
                                  default_export_dir=str(DEFAULT_EXPORT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M")))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>U-Bahn Plan Matcher</title>
<link rel="stylesheet" href="/static/review.css">
<style>
  :root {
    --bg: #eef1f5; --card: #fff; --text: #14213d; --muted: #66738a;
    --line: #dde3ec; --blue: #0a4a94; --blue-dark: #062d5e; --yellow: #ffd400;
    --ok: #157347; --bad: #b42318; --soft: #f4f7fb;
    --u1:#7dad4c; --u2:#da421e; --u3:#007a5b; --u4:#f0d722; --u5:#7e5330;
    --u6:#8c6dab; --u7:#528dba; --u8:#224f86; --u9:#f3791d; --u55:#7e5330;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0c1424; --card: #131d33; --text: #e8edf6; --muted: #93a1ba;
      --line: #24314d; --blue: #2f7fd6; --blue-dark: #0a3a78; --soft: #0f1a2e;
      --ok: #4cc38a; --bad: #f97066;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 16px/1.5 "Helvetica Neue", Arial, system-ui, sans-serif;
  }

  /* Kopfleiste wie ein Bahnsteig-Schild */
  header {
    background: var(--blue-dark); color: #fff; border-bottom: 6px solid var(--yellow);
  }
  .bar { max-width: 1120px; margin: 0 auto; padding: 22px 16px; display: flex; align-items: center; gap: 16px; }
  .logo {
    flex: none; width: 52px; height: 52px; border-radius: 8px; background: #fff; color: var(--blue);
    display: grid; place-items: center; font: 900 2.2rem/1 Arial, sans-serif;
    box-shadow: inset 0 0 0 4px var(--blue);
  }
  header h1 { font-size: 1.5rem; margin: 0; letter-spacing: .01em; }
  header p { margin: 2px 0 0; color: #b9cbea; font-size: .95rem; }
  .lines { margin-left: auto; display: flex; gap: 5px; flex-wrap: wrap; justify-content: flex-end; }
  @media (max-width: 720px) { .lines { display: none; } }

  main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 64px; }
  .grid { display: grid; grid-template-columns: 5fr 6fr; gap: 24px; align-items: start; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px; box-shadow: 0 6px 20px rgba(6, 45, 94, .07);
  }
  h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin: 0 0 14px; }

  /* Linien-Kennzeichen */
  .chip {
    display: inline-block; min-width: 2.1em; padding: 0 .45em; border-radius: 4px; text-align: center;
    font-weight: 800; font-size: .85em; line-height: 1.6; color: #fff; background: var(--blue);
  }
  .chip.u1 { background: var(--u1); } .chip.u2 { background: var(--u2); } .chip.u3 { background: var(--u3); }
  .chip.u4 { background: var(--u4); color: #222; } .chip.u5 { background: var(--u5); }
  .chip.u55 { background: var(--u55); } .chip.u6 { background: var(--u6); } .chip.u7 { background: var(--u7); }
  .chip.u8 { background: var(--u8); } .chip.u9 { background: var(--u9); }

  .drop {
    display: block; border: 2px dashed var(--line); border-radius: 12px; padding: 30px 16px;
    text-align: center; color: var(--muted); cursor: pointer; transition: .15s; background: var(--soft);
  }
  .drop.over, .drop:hover { border-color: var(--blue); background: color-mix(in srgb, var(--blue) 8%, var(--card)); }
  .drop .ico { font-size: 2rem; line-height: 1; color: var(--blue); }
  .drop strong { display: block; color: var(--text); margin-top: 6px; }
  #preview {
    display: none; cursor: zoom-in; background: #fff; border: 1px solid var(--line);
    max-width: 100%; max-height: 420px; margin: 16px auto 0; border-radius: 8px;
  }
  .row { display: flex; align-items: center; gap: 10px; margin: 16px 0; color: var(--muted); font-size: .95rem; }
  .row input { accent-color: var(--blue); width: 18px; height: 18px; }
  button {
    width: 100%; padding: 13px; border: 0; border-radius: 10px; cursor: pointer;
    background: var(--yellow); color: #14213d; font-size: 1rem; font-weight: 800; letter-spacing: .02em;
    box-shadow: 0 3px 0 #c9a600; transition: transform .05s;
  }
  button:active { transform: translateY(2px); box-shadow: 0 1px 0 #c9a600; }
  button:disabled { opacity: .45; cursor: not-allowed; box-shadow: none; }
  .hint { font-size: .85em; opacity: .8; }
  .empty { color: var(--muted); }
  .error { color: var(--bad); }

  /* Ergebnis: Stationsschild */
  .sign {
    background: var(--blue-dark); color: #fff; border-radius: 10px; padding: 18px 20px;
    border-bottom: 6px solid var(--yellow); margin-bottom: 8px;
  }
  .sign .small { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #9fb7dd; }
  .sign .name { font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin: 2px 0 10px; }
  .sign .chip { font-size: 1rem; margin-right: 4px; }
  .field { padding: 12px 0; border-top: 1px solid var(--line); }
  .field:first-of-type { border-top: 0; }
  .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: 2px; }
  .badge { display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: .85rem; font-weight: 700; }
  .high { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
  .medium { background: color-mix(in srgb, #d9a400 24%, transparent); color: #9a7400; }
  .low { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
  .spinner {
    display: inline-block; width: 16px; height: 16px; margin-right: 8px; vertical-align: -3px;
    border: 2px solid rgba(20,33,61,.3); border-top-color: #14213d; border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  details { margin-top: 14px; color: var(--muted); }
  pre { white-space: pre-wrap; font: .85rem/1.4 ui-monospace, Menlo, monospace; }

  .right { min-width: 0; }
  .net { margin-top: 0; padding: 10px 10px 8px; border-radius: 14px; background: #fff; border: 1px solid var(--line); box-shadow: 0 6px 20px rgba(6, 45, 94, .07); }
  .net[hidden], #setup[hidden], #running[hidden] { display: none; }
  #running .ghost { margin-top: 14px; }
  #thumbnote { margin-top: 6px; }
  .net svg { display: block; width: 100%; height: auto; }
  .netcap { color: #14213d; text-align: center; font-size: .9rem; margin-top: 4px; }
  .netcap .chip { margin-right: 5px; }
  .picked { margin-top: 10px; color: var(--muted); font-size: .9rem; text-align: center; min-height: 1.3em; }
  .now { margin-top: 16px; }
  .now.on { display: block; }
  .now .fn { font-weight: 700; word-break: break-all; }
  .bar2 { height: 8px; background: var(--soft); border: 1px solid var(--line); border-radius: 99px; overflow: hidden; margin: 8px 0; }
  .bar2 > div { height: 100%; width: 0; background: var(--yellow); transition: width .3s; }
  #thumb { display: block; max-width: 100%; max-height: 360px; margin-top: 8px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
  #thumb[hidden] { display: none; }
  .btns { display: flex; gap: 10px; }
  .btns button { flex: 1; }
  .btns button[hidden] { display: none; }
  button.ghost { background: var(--card); color: var(--text); border: 1px solid var(--line); box-shadow: none; font-weight: 600; }
  button.ghost:active { transform: none; }
  table { width: 100%; border-collapse: collapse; font-size: .95rem; }
  th { text-align: left; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); padding: 0 8px 8px; }
  td { padding: 10px 8px; border-top: 1px solid var(--line); vertical-align: top; }
  td.doc { word-break: break-all; }
  tr.clickable { cursor: pointer; }
  tr.clickable:hover td { background: var(--soft); }
  .chev { display: inline-block; width: 1.1em; color: var(--muted); }
  tr.detail td { background: var(--soft); border-top: 0; padding: 12px 16px 14px 30px; cursor: default; }
  .rf { margin-bottom: 8px; word-break: normal; }
  .rf .label { margin-bottom: 0; }
  tr.run td { background: color-mix(in srgb, var(--yellow) 22%, transparent); }
  tr.wait td, tr.skip td { color: var(--muted); }
  .none { color: var(--muted); font-style: italic; }
  .tools { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  details.settings { margin: 14px 0 4px; border: 1px solid var(--line); border-radius: 10px; padding: 0 14px; background: var(--soft); }
  details.settings summary { cursor: pointer; padding: 10px 0; font-weight: 600; font-size: .92rem; color: var(--muted); list-style-position: inside; }
  details.settings[open] summary { color: var(--text); }
  details.settings .row:last-child { margin-bottom: 14px; }
  details.settings select { font: inherit; font-size: .9rem; padding: 3px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--text); }
  .gt { flex-wrap: wrap; gap: 6px 10px; }
  .gt input[type=file] { font-size: .85rem; max-width: 100%; color: var(--muted); }
  .gt select { font: inherit; font-size: .9rem; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--line);
               background: var(--card); color: var(--text); max-width: 100%; }
  .gt input[type=file][hidden] { display: none; }
  .tools h2 { margin: 0; }
  .tools button { width: auto; padding: 5px 12px; font-size: .85rem; }
  .tools button[hidden] { display: none; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <div class="logo">U</div>
    <div>
      <h1>U-Bahn Plan Matcher</h1>
      <p>Select a folder of construction plans – {{ model }} maps each document to its Berlin U-Bahn station.</p>
    </div>
    <div class="lines">
      {% for l in ["U1","U2","U3","U4","U5","U6","U7","U8","U9"] %}<span class="chip {{ l|lower }}">{{ l }}</span>{% endfor %}
    </div>
  </div>
</header>
<main>
  <div class="grid">
    <section class="card">
      <div id="setup">
      <label class="drop" id="drop">
        <input type="file" id="files" webkitdirectory multiple hidden>
        <div class="ico">🚇</div><strong>Choose a folder with plans</strong>
        <div id="fname">JPG, PNG, TIFF, PDF – subfolders included</div>
      </label>
      <div class="picked" id="picked"></div>
      <details class="settings">
        <summary>Settings</summary>
      <div class="row">
        <input type="checkbox" id="uselist" checked>
        <label for="uselist">Include station list (bhfdaten.csv)</label>
      </div>
      <div class="row">
        <input type="checkbox" id="thinking">
        <label for="thinking">Deep reasoning <span class="hint">(more thorough, about 10× slower)</span></label>
      </div>
      <div class="row gt">
        <label class="gtlabel" for="gtselect">Ground truth</label>
        <select id="gtselect">
          {% for f in gt_files %}<option value="{{ f }}"{% if f == default_gt %} selected{% endif %}>{{ f }}</option>{% endfor %}
          <option value="none">No ground truth (no evaluation)</option>
          <option value="upload">Other file …</option>
        </select>
        <input type="file" id="gtfile" accept=".csv,.txt,text/csv" hidden>
      </div>
      <div class="row">
        <input type="checkbox" id="prep" checked>
        <label for="prep">Preprocessing <span class="hint">(text regions at full resolution)</span></label>
      </div>
      <div class="row">
        <input type="checkbox" id="retry" checked>
        <label for="retry">Second look for "none" answers <span class="hint">(re-checks with deep reasoning)</span></label>
      </div>
      <div class="row">
        <label for="parallel">Plans analyzed in parallel</label>
        <select id="parallel"><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option><option value="4">4</option></select>
      </div>
      <div class="row">
        <input type="checkbox" id="limit">
        <label for="limit">Skip plans that take longer than
          <input type="number" id="limitsec" value="30" min="5" max="600" style="width:4.2em"> s</label>
      </div>
      </details>
      <div class="btns">
        <button id="go" disabled>Analyze folder</button>
      </div>
      </div>

      <div id="running" hidden>
        <div id="net" class="net">
          <svg viewBox="0 0 574 348" role="img" aria-label="Berlin U-Bahn network with a moving train">
            <image href="/netmap" width="574" height="348"/>
          </svg>
          <div id="netcap" class="netcap"></div>
        </div>
        <div class="now" id="now">
          <div class="label" id="nowlabel">Now analyzing</div>
          <div class="fn" id="nowname"></div>
          <div class="bar2"><div id="bar"></div></div>
          <a id="thumblink" target="_blank" title="Open full size"><img id="thumb" alt="Current plan" hidden></a>
          <div class="hint" id="thumbnote" hidden>red: title block · yellow: corner · blue: text regions</div>
        </div>
        <button id="stop" class="ghost">Stop</button>
      </div>
    </section>

    <div class="right">
    <section class="card">
      <div class="tools">
        <h2>Results</h2>
        <div style="display:flex;gap:8px"><button id="folders" class="ghost" hidden>Create folders</button>
          <button id="csv" class="ghost" hidden>Download CSV</button></div>
      </div>
      <div id="restorebar" class="rbar info" hidden></div>
      <div id="reviewbar" class="rbar" hidden></div>
      <div id="out" class="empty">No analysis yet.</div>
    </section>
  </div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const EXT = /\\.(jpe?g|png|gif|webp|tiff?|pdf)$/i;
let files = [], rows = [], current = null, running = false, cancel = false, gtFile = null;

// Ground-Truth-Auswahl: Datei aus dem Projektordner, keine Auswertung oder eigene Datei; wird gemerkt
try {
  const saved = localStorage.getItem("gtChoice");
  if (saved && [...$("gtselect").options].some(o => o.value === saved && saved !== "upload")) $("gtselect").value = saved;
} catch (e) { /* localStorage kann blockiert sein */ }
$("gtselect").onchange = () => {
  const v = $("gtselect").value;
  $("gtfile").hidden = v !== "upload";
  if (v === "upload") $("gtfile").click();
  try { if (v !== "upload") localStorage.setItem("gtChoice", v); } catch (e) {}
};
$("gtfile").onchange = e => { gtFile = e.target.files[0] || null; };

// ---- Folder selection ---------------------------------------------------------
const path = f => f.webkitRelativePath || f.name;

function setFiles(list) {
  list = [...list];
  files = list.filter(f => EXT.test(f.name) && !f.name.startsWith("."))
    .sort((a, b) => path(a).localeCompare(path(b), undefined, {numeric: true}));
  const skipped = list.length - files.length;
  $("picked").textContent = files.length
    ? `${files.length} plan${files.length > 1 ? "s" : ""} found` + (skipped ? ` (${skipped} other files ignored)` : "")
    : "No supported plans found.";
  $("go").disabled = !files.length || running;
}

$("files").onchange = e => setFiles(e.target.files);
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async e => {
  // Dropped folders are read recursively
  const entries = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  const found = [];
  const walk = async (en, prefix) => {
    if (en.isFile) {
      const f = await new Promise(r => en.file(r));
      Object.defineProperty(f, "webkitRelativePath", {value: prefix + en.name});
      found.push(f);
    } else {
      const rd = en.createReader();
      let batch;
      do {
        batch = await new Promise(r => rd.readEntries(r));
        for (const c of batch) await walk(c, prefix + en.name + "/");
      } while (batch.length);
    }
  };
  for (const en of entries) await walk(en, "");
  setFiles(found);
});

// ---- Results table ------------------------------------------------------------
function parse(text) {
  const get = label => {
    const m = text.match(new RegExp("^[\\\\s*\\\\-]*\\\\*\\\\*" + label + ":\\\\*\\\\*\\\\s*(.*)$", "mi"));
    return m ? m[1].replace(/\\*/g, "").trim() : "";
  };
  let station = get("Station").replace(/^U-Bahnhof\\s+/i, "");
  if (/^(none|n\\/a|not\\b|unknown|unclear|no station)/i.test(station)) station = "";
  const c = get("Confidence"), lvl = (c.match(/^\\s*(high|medium|low)/i) || [])[1];
  return {station, level: lvl ? lvl.toLowerCase() : ""};
}

function renderTable() {
  if (!rows.length) return;
  $("out").className = "";
  $("out").innerHTML = `<table><thead><tr><th>Document</th><th>Station</th><th>Confidence</th></tr></thead><tbody>` +
    rows.map((r, i) => {
      let st, cf;
      if (r.state === "wait") st = cf = "queued";
      else if (r.state === "run") st = cf = "analyzing …";
      else if (r.state === "skip") { st = `<span class="none">${esc(r.error)}</span>`; cf = "–"; }
      else if (r.state === "err") { st = `<span class="error">${esc(r.error)}</span>`; cf = "–"; }
      else {
        st = r.station ? `<b>${esc(r.station)}</b>` : `<span class="none">no station assigned</span>`;
        cf = r.level ? `<span class="badge ${r.level}">${r.level}</span>` : "–";
      }
      const sc = r.scaled ? `<div class="hint">downscaled ${r.scaled.from.join("×")} → ${r.scaled.to.join("×")} px</div>` : "";
      const labels = {spelling: "spelling check", date: "date rule", line: "line check", "line-flag": "line check"};
      const chk = (r.corr || []).map(c => `<div class="hint">${esc(labels[c.type] || "check")}: ${
        esc(c.type === "line-flag" ? "plan line does not fit " + c.from + " (low confidence)" : c.from + " → " + c.to)}</div>`).join("");
      const sl = chk + (r.retried ? `<div class="hint">second look with deep reasoning</div>` : "") +
        (r.prep && r.prep.title_block ? `<div class="hint">text regions: title block + ${r.prep.regions} more</div>` :
         r.prep && r.prep.disabled ? "" :
         r.prep && !r.prep.error ? `<div class="hint">no text regions found, whole sheet sent</div>` :
         r.prep ? `<div class="hint error">preprocessing failed, whole sheet sent</div>` : "");
      if (typeof reviewDecorate === "function") ({st, cf} = reviewDecorate(r, st, cf));
      const extra = typeof reviewDocExtra === "function" ? reviewDocExtra(r, i) : "";
      const open = r.open && r.answer;
      const chev = r.answer ? `<span class="chev">${open ? "▾" : "▸"}</span>` : "";
      return `<tr class="${r.state}${r.answer ? " clickable" : ""}" data-i="${i}"><td class="doc">${chev}${esc(r.title)}${extra}${sc}${sl}</td><td>${st}</td><td>${cf}</td></tr>` +
        (open ? `<tr class="detail"><td colspan="3">${reasoningHtml(r.answer)}</td></tr>` : "");
    }).join("") + "</tbody></table>";
}

// Begründung des Modells (alle Felder der Antwort) für das Klappfeld unter einem Plan
function reasoningHtml(answer) {
  const re = /^[\\s*\\-]*\\*\\*(.+?):\\*\\*\\s*(.*)$/;
  const fields = [];
  for (const line of answer.split("\\n")) {
    const m = line.match(re);
    if (m) fields.push({label: m[1], value: m[2]});
    else if (fields.length && line.trim()) fields[fields.length - 1].value += "\\n" + line;
  }
  if (!fields.length) return `<pre>${esc(answer)}</pre>`;
  return fields.map(f => `<div class="rf"><div class="label">${esc(f.label)}</div><div>${
    esc(f.value.replace(/\\*\\*/g, "")).replace(/\\n/g, "<br>")}</div></div>`).join("");
}

// Klick auf einen analysierten Plan klappt die Begründung auf oder zu
$("out").addEventListener("click", e => {
  const tr = e.target.closest("tr[data-i]");
  const r = tr && rows[Number(tr.dataset.i)];
  if (!r || !r.answer) return;
  r.open = !r.open;
  renderTable();
});

$("csv").onclick = () => {
  const q = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const fin = r => typeof effectiveStation === "function" ? effectiveStation(r) : r.station;
  const lines = [["document", "model_station", "confidence", "final_station", "decided_by", "downscaled", "error"].map(q).join(",")]
    .concat(rows.map(r => [r.title, r.station, r.level, fin(r), typeof decidedBy === "function" ? decidedBy(r) : "",
      r.scaled ? r.scaled.from.join("x") + " -> " + r.scaled.to.join("x") : "", r.error].map(q).join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\\ufeff" + lines.join("\\n")], {type: "text/csv;charset=utf-8"}));
  a.download = "ubahn_plan_mapping.csv";
  a.click();
};

// ---- Preview of the plan currently being analyzed -----------------------------
let thumbUrl = null;
async function showThumb(r) {
  const f = r.file;
  const img = $("thumb");
  img.hidden = true;
  if (thumbUrl) URL.revokeObjectURL(thumbUrl);
  thumbUrl = null;
  try {
    let url;
    if (/^image\\/(jpeg|png|gif|webp)$/.test(f.type)) url = URL.createObjectURL(f);
    else {
      const fd = new FormData();
      fd.append("image", f);
      if (r.page) fd.append("page", r.page);
      const res = await fetch("/preview", {method: "POST", body: fd});
      if (!res.ok) return;
      url = URL.createObjectURL(await res.blob());
    }
    if (current !== r) { URL.revokeObjectURL(url); return; }  // already on another plan
    thumbUrl = url;
    img.src = $("thumblink").href = url;
    img.hidden = false;
  } catch (e) { /* the preview is optional */ }
}

// Vorschau mit den erkannten Textregionen (Debug-Overlay des Preprocessings); sonst der Plan selbst
async function showOverlay(r, prepOn) {
  const f = r.file;
  const started = Date.now();
  if (prepOn) {
    const img = $("thumb");
    img.hidden = true; $("thumbnote").hidden = true;
    if (thumbUrl) URL.revokeObjectURL(thumbUrl);
    thumbUrl = null;
    try {
      const fd = new FormData();
      fd.append("image", f);
      if (r.page) fd.append("page", r.page);
      const res = await fetch("/preprocess", {method: "POST", body: fd});
      if (res.ok) {
        const url = URL.createObjectURL(await res.blob());
        if (current !== r) { URL.revokeObjectURL(url); return (Date.now() - started) / 1000; }
        thumbUrl = url;
        img.src = $("thumblink").href = url;
        img.hidden = false; $("thumbnote").hidden = false;
        return (Date.now() - started) / 1000;
      }
    } catch (e) { /* weiter mit der normalen Vorschau */ }
  }
  $("thumbnote").hidden = true;
  await showThumb(r);
  return prepOn ? (Date.now() - started) / 1000 : 0;
}

// --- Ladeanimation: Zug fährt die Linien des Netzplans (ubahnlanes.png) ab -----
// Koordinaten in Pixeln des Bildes (574 x 348), von Hand nachgezeichnet.
const NET = [
  {n: "U1", c: "#7dad4c", p: [[177,209],[404,209]]},
  {n: "U2", c: "#da421e", p: [[76,155],[100,181],[106,184],[190,184],[200,188],[203,191],[220,205],[237,207],[255,215],[266,226],[272,221],[272,187],[277,184],[317,183],[369,136],[369,88]]},
  {n: "U3", c: "#007a5b", p: [[102,326],[212,216],[218,215],[404,215]]},
  {n: "U4", c: "#f0d722", p: [[237,207],[237,277]]},
  {n: "U5", c: "#7e5330", p: [[256,154],[462,154],[478,170],[486,178],[500,178],[556,117]]},
  {n: "U6", c: "#8c6dab", p: [[195,33],[230,68],[262,100],[310,150],[311,326]]},
  {n: "U7", c: "#528dba", p: [[27,141],[142,141],[150,148],[150,237],[160,250],[250,250],[257,250],[287,218],[291,217],[310,235],[318,241],[365,242],[375,246],[459,333]]},
  {n: "U8", c: "#224f86", p: [[247,10],[247,40],[350,140],[351,150],[351,281]]},
  {n: "U9", c: "#f3791d", p: [[283,73],[262,99],[202,158],[200,163],[200,290],[183,311]]},
];
const SVGNS = "http://www.w3.org/2000/svg";
let netRaf = null, netBuilt = false;

function svgEl(name, attrs, parent) {
  const el = document.createElementNS(SVGNS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  parent.appendChild(el);
  return el;
}

function buildNet(svg) {
  NET.forEach(l => {
    // unsichtbarer Pfad dient nur zum Abfahren; sichtbar wird er nur als Hervorhebung
    l.path = svgEl("path", {
      d: "M" + l.p.map(q => q.join(",")).join(" L"), fill: "none", stroke: l.c, opacity: 0,
      "stroke-width": 5, "stroke-linecap": "round", "stroke-linejoin": "round",
    }, svg);
    l.len = l.path.getTotalLength();
  });
  const t = svgEl("g", {}, svg);
  svgEl("rect", {x: -13, y: -6, width: 26, height: 12, rx: 4, fill: "#fff", stroke: "#0b1b34", "stroke-width": 1.6}, t);
  svgEl("rect", {x: 4, y: -4, width: 6, height: 8, rx: 1.5, fill: "#0b1b34"}, t);    // Führerstand
  svgEl("rect", {x: -10, y: -3, width: 4, height: 6, rx: 1, fill: "#0b1b34"}, t);   // Fenster
  svgEl("rect", {x: -4, y: -3, width: 4, height: 6, rx: 1, fill: "#0b1b34"}, t);
  svgEl("rect", {x: -13, y: 3, width: 26, height: 3, rx: 1, fill: "#ffd400"}, t);    // gelber Streifen
  svg._train = t;
  netBuilt = true;
}

function startNet() {
  const box = $("net"), svg = box.querySelector("svg");
  box.hidden = false;
  if (!netBuilt) buildNet(svg);
  const cap = $("netcap"), train = svg._train, SPEED = 130;  // Bildpunkte pro Sekunde
  let i = 0, pos = 0, last = null;

  const activate = k => {
    NET.forEach((l, j) => l.path.setAttribute("opacity", j === k ? .9 : 0));
    svg.appendChild(train);  // Zug über die Hervorhebung legen
    cap.innerHTML = `<span class="chip ${NET[k].n.toLowerCase()}">${NET[k].n}</span>Train running on ${NET[k].n} …`;
  };
  activate(0);

  const step = ts => {
    if (last === null) last = ts;
    pos += Math.min(ts - last, 100) / 1000 * SPEED;
    last = ts;
    if (pos > NET[i].len) { i = (i + 1) % NET.length; pos = 0; activate(i); }
    const cur = NET[i], at = Math.min(pos, cur.len);
    const a = cur.path.getPointAtLength(at), b = cur.path.getPointAtLength(Math.min(at + 2, cur.len));
    const ang = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
    train.setAttribute("transform", `translate(${a.x} ${a.y}) rotate(${ang})`);
    netRaf = requestAnimationFrame(step);
  };
  netRaf = requestAnimationFrame(step);
}

function stopNet() {
  if (netRaf) cancelAnimationFrame(netRaf);
  netRaf = null;
  $("net").hidden = true;
}

// ---- Run ------------------------------------------------------------------------
$("stop").onclick = () => {
  cancel = true;
  $("stop").disabled = true;
  $("nowlabel").textContent = "Stopping after the current plan …";
};

// A PDF with several pages becomes one row (and one analysis) per page
async function expandPages(list) {
  const out = [];
  for (const f of list) {
    const base = {file: f, state: "wait", station: "", level: "", error: ""};
    let pages = 1;
    if (f.name.toLowerCase().endsWith(".pdf")) {
      try {
        const fd = new FormData();
        fd.append("image", f);
        const res = await fetch("/pdf_pages", {method: "POST", body: fd});
        if (res.ok) pages = Math.min((await res.json()).pages || 1, 200);
      } catch (e) { /* treat as a single page */ }
    }
    if (pages > 1) for (let p = 1; p <= pages; p++) out.push(Object.assign({}, base, {page: p, title: `${path(f)} (page ${p})`}));
    else out.push(Object.assign({}, base, {title: path(f)}));
  }
  return out;
}

$("go").onclick = async () => {
  running = true; cancel = false;
  $("go").disabled = true;
  $("stop").disabled = false;
  $("csv").hidden = true;
  $("setup").hidden = true; $("running").hidden = false;  // Auswahl ausblenden, Ladeansicht zeigen
  $("nowlabel").textContent = "Preparing …"; $("nowname").textContent = ""; $("bar").style.width = "0%";
  startNet();
  rows = await expandPages(files);
  renderTable();
  const use = $("uselist").checked ? "1" : "0", think = $("thinking").checked ? "1" : "0";
  const prepOn = $("prep").checked;
  const limitSec = $("limit").checked ? Math.max(5, parseInt($("limitsec").value) || 30) : 0;
  const par = Math.max(1, Math.min(4, parseInt($("parallel").value) || 1));
  const started = Date.now();
  let next = 0, finished = 0;
  const active = new Set();

  const progress = () => {
    $("bar").style.width = (finished / rows.length * 100) + "%";
    let eta = "";
    if (finished >= 2 && finished < rows.length) {
      const left = (Date.now() - started) / finished * (rows.length - finished) / 1000;
      eta = left < 90 ? ` · about ${Math.max(5, Math.round(left / 5) * 5)} s left` : ` · about ${Math.round(left / 60)} min left`;
    }
    $("nowlabel").textContent = cancel ? "Stopping after the running plans …" : `Analyzing – ${finished} of ${rows.length} done${eta}`;
    $("nowname").textContent = [...active].slice(-3).map(x => x.title).join("  ·  ");
  };

  const work = async r => {
    r.state = "run";
    active.add(r);
    renderTable(); progress();
    current = r;
    try {
      const usedSec = await showOverlay(r, prepOn);
      const fd = new FormData();
      fd.append("image", r.file);
      if (r.page) fd.append("page", r.page);
      fd.append("use_list", use);
      fd.append("thinking", think);
      fd.append("retry", $("retry").checked ? "1" : "0");
      fd.append("time_limit", limitSec);
      fd.append("preprocess", prepOn ? "1" : "0");
      fd.append("time_used", usedSec.toFixed(1));
      const res = await fetch("/analyze", {method: "POST", body: fd});
      const raw = await res.text();
      let data;
      try { data = JSON.parse(raw); } catch (e) { throw new Error(`Server error ${res.status}: ${raw.slice(0, 120)}`); }
      if (!res.ok) throw new Error(data.error || res.statusText);
      if (data.skipped) { r.state = "skip"; r.error = data.reason; r.prep = data.preprocess || null; r.plan_id = data.plan_id; return; }
      Object.assign(r, parse(data.result), typeof parseExtraFields === "function" ? parseExtraFields(data.result) : {},
        {state: "done", scaled: data.scaled || null, retried: !!data.retried, prep: data.preprocess || null,
         corr: data.corrections || [], answer: data.result, plan_id: data.plan_id});
      if (r.retried && r.level === "high") r.level = "medium";  // answer needed a second look: not fully certain
    } catch (err) {
      r.state = "err";
      r.error = err.message;
    } finally {
      active.delete(r);
      finished++;
      renderTable(); progress();
    }
  };
  const worker = async () => {
    while (!cancel) {
      const i = next++;
      if (i >= rows.length) return;
      await work(rows[i]);
    }
  };
  progress();
  await Promise.all(Array.from({length: par}, worker));
  rows.forEach(r => { if (r.state === "wait" || r.state === "run") { r.state = "err"; r.error = "not analyzed"; } });
  renderTable();
  current = null;
  $("bar").style.width = "100%";
  $("nowlabel").textContent = cancel ? "Stopped" : "Done";
  $("nowname").textContent = "";
  $("thumb").hidden = true; $("thumbnote").hidden = true;
  stopNet();
  running = false;
  $("running").hidden = true; $("setup").hidden = false;  // Auswahl wieder zeigen
  $("go").disabled = !files.length;
  $("csv").hidden = false;
  if (typeof afterRun === "function") afterRun();
  if (rows.some(r => r.state === "done")) {
    await buildReport({thinking: think === "1", station_list: use === "1", seconds: Math.round((Date.now() - started) / 1000),
                       stopped: cancel, plans_in_folder: rows.length,
                       time_limit: limitSec, preprocessing: prepOn, skipped: rows.filter(r => r.state === "skip").length});
  }
};

// ---- Report: saved on the server (reports/report_<id>_<time>.json / .csv), not shown in the page ---------
async function buildReport(meta) {
  try {
    const fd = new FormData();
    const choice = $("gtselect").value;
    fd.append("gt_choice", choice);
    if (choice === "upload" && gtFile) fd.append("ground_truth", gtFile);
    fd.append("rows", JSON.stringify(rows.filter(r => r.state !== "wait").map(r =>
      ({title: r.title, station: r.station, level: r.level, scaled: r.scaled || null, retried: !!r.retried, prep: r.prep || null, corr: r.corr || [],
        answer: r.answer || "", error: r.state === "err" || r.state === "skip" ? r.error : ""}))));
    fd.append("meta", JSON.stringify(meta));
    const res = await fetch("/report", {method: "POST", body: fd});
    if (!res.ok) console.warn("The report could not be saved:", (await res.text()).slice(0, 200));
  } catch (err) {
    console.warn("The report could not be saved:", err);
  }
}
</script>
<script>window.STATION_NAMES = {{ station_names|tojson }}; window.DEFAULT_EXPORT_DIR = {{ default_export_dir|tojson }};</script>
<script src="/static/review.js"></script>
</body>
</html>
"""

if __name__ == "__main__":
    # Port 5000 belegt unter macOS der AirPlay-Empfänger (antwortet im Browser mit "403 Forbidden"),
    # daher standardmäßig 5050; mit der Umgebungsvariable PORT änderbar.
    port = int(os.environ.get("PORT", 5050))
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
