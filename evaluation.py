"""Auswertung der Zuordnungen gegen eine Ground Truth.

Ground Truth: CSV (Trennzeichen ; , oder Tab) mit einer Zeile pro Plan.
  - Kopfzeile mit Spalte "...station..."/"...bahnhof..." und Spalte "image"/"file"/"datei"/"document"/"plan"
    (Reihenfolge egal); ohne Kopfzeile: Spalte 1 = Datei, Spalte 2 = Station
  - Station: aktueller Name; leer, "none", "no station", "not identified" = keine Station
Der Abgleich der Dateien läuft über den vollen Pfad oder den Dateinamen (Unicode-normalisiert, ohne
Groß-/Kleinschreibung, Endung egal; als letzter Ausweg bei fast gleichem Namen).
Stationsnamen werden vor dem Vergleich normalisiert und über bhfdaten.csv (Spalte "heute") auf den
aktuellen Namen abgebildet.
"""
import csv
import difflib
import io
import re
import unicodedata
from collections import Counter, defaultdict

# Umbenennungen, die in der Stationsliste (noch) nicht nachgezogen sind
EXTRA_ALIASES = {
    "Mohrenstraße": "Anton-Wilhelm-Amo-Straße",  # umbenannt 2025
    "Oisstraße": "Otisstraße",                    # Tippfehler in der Liste
}
NONE_WORDS = {"", "none", "n/a", "na", "-", "–", "keine", "kein bahnhof", "no station",
              "not identified", "nicht identifiziert", "unknown", "unbekannt"}


def normalize(name):
    """Vergleichsschlüssel: ohne Präfix/Klammern, klein, ß=ss, 'str.'='strasse'."""
    name = re.sub(r"\[.*?\]|\(.*?\)", "", name or "")
    name = re.sub(r"^(U-Bahnhof|Bahnhof|U)\s+", "", name.strip(), flags=re.I)
    name = name.lower().replace("ß", "ss").replace("str.", "strasse")
    return re.sub(r"[\s\-]", "", name)


class StationNames:
    """Bildet beliebige (auch frühere) Namen auf den aktuellen Namen ab."""

    def __init__(self, stations_csv=None):
        self.current = {}  # Schlüssel -> aktueller Anzeigename
        self.station_names = []  # alle Namen der Stationsliste
        self.known_keys, self.known_keys_display = [], {}
        if stations_csv:
            try:
                with open(stations_csv, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except OSError:
                rows = []
            self.station_names = [r["name"] for r in rows]
            for r in rows:
                if not r.get("heute"):
                    k = normalize(r["name"])
                    if k not in self.known_keys_display:
                        self.known_keys.append(k)
                        self.known_keys_display[k] = re.sub(r"\s*\[.*?\]", "", r["name"]).strip()
            # Ein Name, der zugleich eine aktuelle Station ist (z. B. "Friedrichstraße": früherer Name
            # von Stadtmitte, aber auch heutiger U6-Bahnhof), wird nicht umgemappt.
            is_current = {normalize(r["name"]) for r in rows if not r.get("heute")}
            for r in rows:
                key = normalize(r["name"])
                if r.get("heute") and key not in is_current:
                    self.current[key] = r["heute"]
        for old, new in EXTRA_ALIASES.items():
            self.current[normalize(old)] = new
        # Auch die Ziel-Namen von Umbenennungen sind gültige Stationsnamen (z. B. "Otisstraße", das in
        # der Liste noch als "Oisstraße" steht); sonst würde das Einrasten sie wieder zurücksetzen.
        for target in self.current.values():
            k = normalize(target)
            if k not in self.known_keys_display:
                self.known_keys.append(k)
                self.known_keys_display[k] = re.sub(r"\s*\[.*?\]", "", target).strip()

    def canonical(self, name, snap=False):
        """Gibt (schlüssel, anzeigename) des aktuellen Namens zurück oder (None, None)."""
        if name is None or name.strip().lower() in NONE_WORDS:
            return None, None
        display = re.sub(r"^U-Bahnhof\s+", "", name.strip(), flags=re.I)
        display = re.sub(r"\s*\[.*?\]", "", display).strip()  # "Stadtmitte [oben]" -> "Stadtmitte"
        seen = set()
        key = normalize(display)
        while key in self.current and key not in seen:  # Ketten wie Kaiserhof -> Mohrenstraße -> Amo
            seen.add(key)
            display = re.sub(r"\s*\[.*?\]", "", self.current[key]).strip()
            key = normalize(display)
        if snap and key not in self.known_keys:  # Modell hat den Namen leicht falsch geschrieben
            close = difflib.get_close_matches(key, self.known_keys, 1, 0.9)
            if close:
                key = close[0]
                display = self.known_keys_display[key]
        return key, display


def _key(text):
    return unicodedata.normalize("NFC", text).replace("\\", "/").strip().casefold()


def load_ground_truth(data):
    """CSV-Bytes -> {dateiname_schlüssel: stationsname_oder_leer}."""
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text.split("\n", 1)[0] + "\n", delimiters=";,\t")
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return {}
    head = [h.strip().lower() for h in rows[0]]
    st_i = next((i for i, h in enumerate(head) if "station" in h or "bahnhof" in h), None)
    doc_i = next((i for i, h in enumerate(head)
                  if i != st_i and any(w in h for w in ("image", "file", "datei", "document", "plan", "name"))), None)
    if st_i is not None and doc_i is not None:
        start = 1
    else:
        doc_i, st_i, start = 0, 1, 0
    truth = {}
    for r in rows[start:]:
        if len(r) > max(doc_i, st_i) and r[doc_i].strip():
            truth[_key(r[doc_i])] = r[st_i].strip()
    return truth


def _stem(key):
    base = key.rsplit("/", 1)[-1]
    return re.sub(r"\.[a-z0-9]{2,5}$", "", base)


def _lookup(truth, title, used):
    """Sucht den Ground-Truth-Eintrag zu einem Plan; gibt den Schlüssel oder None zurück."""
    t = _key(title)
    if t in truth:
        return t
    base = t.rsplit("/", 1)[-1]
    same_base = [k for k in truth if k.rsplit("/", 1)[-1] == base]
    if same_base:
        return same_base[0]
    stem = _stem(t)
    same_stem = [k for k in truth if _stem(k) == stem]
    if same_stem:
        return same_stem[0]
    free = {_stem(k): k for k in truth if k not in used}  # fast gleicher Name, z. B. Tippfehler
    close = difflib.get_close_matches(stem, list(free), 1, 0.9)
    return free[close[0]] if close else None


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def evaluate(rows, truth, names):
    """rows: [{title, station, level, error}]; truth: load_ground_truth(); names: StationNames.

    Ein Plan ohne Vorhersage (leer) zählt als "keine Station". Pläne, die beim Analysieren
    einen Fehler hatten, werden getrennt gemeldet und nicht mitgezählt.
    """
    items, no_truth, errors = [], [], []
    used = set()
    for r in rows:
        if r.get("error"):
            errors.append({"document": r["title"], "error": r["error"]})
            continue
        k = _lookup(truth, r["title"], used)
        if k is None:
            no_truth.append(r["title"])
            continue
        used.add(k)
        gk, gd = names.canonical(truth[k])
        pk, pd = names.canonical(r.get("station"), snap=True)
        items.append({"document": r["title"], "truth": gd, "pred": pd, "truth_key": gk, "pred_key": pk,
                      "level": r.get("level") or "unknown", "correct": gk == pk})

    known = {normalize(n) for n in names.station_names} | {normalize(v) for v in names.current.values()}
    unknown_truth = sorted({truth[k] for k in used if names.canonical(truth[k])[0] not in (None, *known)})

    n = len(items)
    correct = sum(i["correct"] for i in items)
    should = [i for i in items if i["truth_key"]]
    assigned = [i for i in items if i["pred_key"]]
    tp = sum(1 for i in items if i["truth_key"] and i["correct"])
    fp = len(assigned) - tp                       # zugeordnet, aber falsch/obwohl keine Station
    fn = len(should) - tp                         # Station vorhanden, aber nicht (richtig) getroffen
    p, r, f = _prf(tp, fp, fn)
    none_items = [i for i in items if not i["truth_key"]]

    # Nach Sicherheit
    by_level = []
    for lvl in ("high", "medium", "low", "unknown"):
        sel = [i for i in items if i["level"] == lvl]
        if sel:
            by_level.append({"level": lvl, "n": len(sel), "correct": sum(i["correct"] for i in sel),
                             "accuracy": sum(i["correct"] for i in sel) / len(sel)})

    # Nach Station (nur echte Stationen)
    tpc, fpc, fnc, disp = Counter(), Counter(), Counter(), {}
    for i in items:
        if i["truth_key"]:
            disp[i["truth_key"]] = i["truth"]
        if i["pred_key"]:
            disp.setdefault(i["pred_key"], i["pred"])
        if i["correct"] and i["truth_key"]:
            tpc[i["truth_key"]] += 1
        else:
            if i["pred_key"]:
                fpc[i["pred_key"]] += 1
            if i["truth_key"]:
                fnc[i["truth_key"]] += 1
    per_station = []
    for k in disp:
        sp, sr, sf = _prf(tpc[k], fpc[k], fnc[k])
        per_station.append({"station": disp[k], "support": tpc[k] + fnc[k], "predicted": tpc[k] + fpc[k],
                            "precision": sp, "recall": sr, "f1": sf})
    per_station.sort(key=lambda s: (-s["support"], s["station"]))
    f1s = [s["f1"] for s in per_station if s["support"] and s["f1"] is not None]

    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "precision": p, "recall": r, "f1": f,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "coverage": len(assigned) / n if n else None,
        "counts": {"correct": correct, "should_assign": len(should), "assigned": len(assigned),
                   "tp": tp, "fp": fp, "fn": fn,
                   "none_total": len(none_items),
                   "none_correct": sum(1 for i in none_items if not i["pred_key"])},
        "by_confidence": by_level,
        "per_station": per_station,
        "wrong": [{k: i[k] for k in ("document", "truth", "pred", "level")} for i in items if not i["correct"]],
        "items": [{k: i[k] for k in ("document", "truth", "pred", "level", "correct")} for i in items],
        "without_ground_truth": no_truth,
        "ground_truth_stations_not_in_list": unknown_truth,
        "ground_truth_unused": sorted(k for k in truth if k not in used),
        "analysis_errors": errors,
    }
