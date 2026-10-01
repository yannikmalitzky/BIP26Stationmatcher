"""Prüfungen der Modellantwort im Code (ohne weiteren Modellaufruf).

Das Modell begründet manchmal Zuordnungen, die der Stationsliste widersprechen. Zwei Prüfungen fangen
das ab; jede Korrektur wird in der Antwort vermerkt und in der Oberfläche angezeigt:

0. Datumsregel (verbindlich): Kann ein Name im Titel zu mehreren Stationen gehören (alte Station umbenannt,
   später neue Station mit dem alten Namen), entscheidet das Plandatum. Ohne Datum gilt die Station, die den
   Namen heute trägt. Die Regel wird im Code durchgesetzt, auch wenn das Modell anders entschieden hat.
1. Schreibweise: Bei einer direkten Zuordnung wird der Titeltext buchstabenweise mit allen Stationsnamen der
   Liste (auch früheren) verglichen. Passt ein anderer Bahnhof deutlich besser als der genannte, wird er
   übernommen (z. B. Titel "Thinsprucker Platz" -> Innsbrucker Platz, nicht die erfundene Herleitung).
2. Linie: Bei einer Zuordnung über die Nähe wird die im Plan genannte Linie ("U5") mit den Linien der
   Station laut Liste verglichen. Passen sie nicht zusammen, wird die Zuordnung auf "none" gesetzt.
"""
import csv
import re
from difflib import SequenceMatcher

from evaluation import StationNames, normalize

MIN_SIMILARITY = 0.80      # so ähnlich muss ein Stationsname dem Titeltext mindestens sein
MIN_GAP = 0.12             # ... und um so viel besser als der vom Modell genannte Bahnhof
KEEP_SIMILARITY = 0.85     # passt der genannte Bahnhof schon so gut, wird nichts geändert
LINE_CHECK_ACTION = "flag"  # "flag": Sicherheit auf "low" setzen und markieren; "none": Station auf "none" setzen
MIN_LEN = 5                # kürzere Wortgruppen werden nicht verglichen
LINE_CHECK_FROM = 1970     # U-Linien-Nummern gibt es erst seit den 1960er/70er Jahren; auf älteren Plänen
                           # bezeichnet "U3" o. ä. andere oder geplante Linien


def _field(answer, label):
    m = re.search(r"^[\s*\-]*\*\*" + label + r":\*\*\s*(.*?)(?=^[\s*\-]*\*\*[A-Za-z ]+:\*\*|\Z)", answer or "",
                  re.M | re.S | re.I)
    return m.group(1).replace("*", "").strip() if m else ""


def _is_none(station):
    return bool(re.match(r"(none|n/a|not\b|unknown|unclear|no station)", station.strip(), re.I)) or not station.strip()


def _replace_field(answer, label, value):
    pat = re.compile(r"^([\s*\-]*\*\*" + label + r":\*\*\s*).*?$", re.M | re.I)
    return pat.sub(lambda m: m.group(1) + value, answer, count=1)


def _year(text):
    """Erstes Jahr in einem Datumstext ("10.10.1990", "1.1.76", "1953"); None, wenn keins zu finden ist."""
    m = re.search(r"\b(1[89]\d\d|20\d\d)\b", text or "")
    if m:
        return int(m.group(1))
    m = re.search(r"\b\d{1,2}\.\d{1,2}\.(\d{2})\b", text or "")
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 30 else 2000 + yy
    return None


def _name_periods(row):
    """Namensperioden einer Station aus "eroeffnung" und "bemerkung": [[Startjahr, {Namensschlüssel}], ...].

    Ein Name in Klammern ("Städtische Oper; (Bismarckstraße)") gilt als weiterhin geführter Zusatzname.
    """
    open_year = _year(row.get("eroeffnung"))
    events = []
    for tok in (t.strip() for t in re.split(r";", row.get("bemerkung") or "")):
        if not tok:
            continue
        m = re.match(r"^(?:eröffnet\s+als|als)\s*:?\s*(.+)$", tok, re.I)
        if m and not events:
            events.append([open_year, {normalize(m.group(1))}])
            continue
        m = re.match(r"^ab\s+(?:etwa\s+)?(?:\d{1,2}\.){0,2}(\d{4})\s*:\s*(.+)$", tok, re.I)
        if m:
            events.append([int(m.group(1)), {normalize(m.group(2))}])
            continue
        if tok.startswith("(") and tok.endswith(")") and ":" not in tok and events:
            events[-1][1].add(normalize(tok[1:-1]))
    return events or [[open_year, {normalize(row["name"])}]]


_GAP = r"\.{2,}|…|\[\s*\.{2,}\s*\]|\?{2,}|_{2,}|x{3,}|\[redacted\]|\[geschwärzt\]"


class PostCheck:
    def __init__(self, stations_csv):
        with open(stations_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names = StationNames(stations_csv)
        self.display = {}    # Schlüssel des aktuellen Namens -> Anzeigename
        self.old_keys = {}   # Schlüssel des aktuellen Namens -> {Schlüssel früherer/abweichender Namen}
        self.lines = {}      # Schlüssel des aktuellen Namens -> Menge von Linien
        for r in rows:
            key, disp = self.names.canonical(r["name"])
            if r.get("heute"):  # früherer Name: gehört (auch bei Namenskonflikten) zu dieser Station
                tkey, tdisp = self.names.canonical(r["heute"])
                self.old_keys.setdefault(tkey, set()).add(normalize(r["name"]))
                self.display.setdefault(tkey, tdisp)
                continue
            self.display.setdefault(key, disp)
            # auch der Name der Zeile selbst (z. B. "Mohrenstraße", inzwischen umbenannt) gehört zur Station
            self.old_keys.setdefault(key, set()).update({key, normalize(r["name"])})
            for ln in re.findall(r"U\d+", r.get("linien") or ""):
                self.lines.setdefault(key, set()).add(ln)
        # Umbenennungen, die die Stationsliste (noch) nicht kennt (siehe evaluation.EXTRA_ALIASES)
        from evaluation import EXTRA_ALIASES
        for old, new in EXTRA_ALIASES.items():
            nkey, ndisp = self.names.canonical(new)
            self.old_keys.setdefault(nkey, set()).update({nkey, normalize(old)})
            self.display.setdefault(nkey, ndisp)
        # Zeiträume, in denen eine Station einen Namen trug: Namensschlüssel -> [(Stationsschlüssel, von, bis)]
        self.name_intervals = {}
        for r in rows:
            if r.get("heute"):
                continue
            skey = self.names.canonical(r["name"])[0]
            events = _name_periods(r)
            for i, (start, keys) in enumerate(events):
                end = events[i + 1][0] if i + 1 < len(events) and events[i + 1][0] else 9999
                for k in keys:
                    lst = self.name_intervals.setdefault(k, [])
                    if lst and lst[-1][0] == skey and lst[-1][2] == (start or 0):  # direkt anschließend: zusammenfassen
                        lst[-1] = (skey, lst[-1][1], end)
                    else:
                        lst.append((skey, start or 0, end))
            self.name_intervals.setdefault(normalize(r["name"]), [])
        # alle Vergleichsschlüssel -> aktuelle Station
        self.key_to_station = {}
        for skey, keys in self.old_keys.items():
            for k in keys:
                self.key_to_station.setdefault(k, skey)

    # -- Schreibweise ---------------------------------------------------------------------------
    @staticmethod
    def _title_text(answer):
        title = _field(answer, "Title")
        quoted = re.search(r'"([^"]+)"', title)
        return quoted.group(1) if quoted else re.sub(r"\([^)]*\)\s*$", "", title)

    @staticmethod
    def _ngrams(text):
        tokens = [normalize(t) for t in re.findall(r"[A-Za-zÄÖÜäöüß]+\.?", text)]
        tokens = [t for t in tokens if t]
        grams = set()
        for n in (1, 2, 3, 4):
            for i in range(len(tokens) - n + 1):
                g = "".join(tokens[i:i + n])
                if len(g) >= MIN_LEN:
                    grams.add(g)
        return grams

    @staticmethod
    def _best(grams, keys):
        best = 0.0
        for k in keys:
            for g in grams:
                if abs(len(g) - len(k)) > max(3, len(k) // 3):
                    continue
                best = max(best, SequenceMatcher(None, g, k).ratio())
        return best

    # -- Datumsregel ----------------------------------------------------------------------------
    def date_rule(self, answer):
        """Gibt (Station, Hinweis) zurück, wenn das Plandatum eine andere Station vorschreibt als das Modell."""
        station = _field(answer, "Station")
        if _is_none(station) or not _field(answer, "Match type").lower().startswith("direct"):
            return None, None
        year = _year(_field(answer, "Plan date"))
        pkey, _ = self.names.canonical(station)
        for gram in sorted(self._ngrams(self._title_text(answer)), key=len, reverse=True):
            intervals = self.name_intervals.get(gram)
            if not intervals or len({sk for sk, _, _ in intervals}) < 2:
                continue  # Name ist eindeutig: keine Datumsentscheidung nötig
            if year is not None:
                hits = {sk for sk, a, b in intervals if a <= year < b}
                why = f"the name '{gram}' was carried by {{}} in {year}"
            else:  # ohne Datum: die Station, die den Namen heute trägt
                hits = {sk for sk, a, b in intervals if b == 9999}
                why = f"the plan has no date, so the station that carries the name '{gram}' today applies ({{}})"
            if len(hits) != 1:
                continue
            target = next(iter(hits))
            if target == pkey:
                return None, None
            new = self.display.get(target, target)
            return new, "date rule: " + why.format(new) + f"; the model had chosen '{station}'"
        return None, None

    # -- Namen mit Lücken (Buchstaben fehlen, Streifen, "...") ----------------------------------
    def _gap_patterns(self, title):
        """Aus einem Titel mit Lücken ("SOPHIE-...-PLATZ") Muster wie 'sophie.*platz' bilden."""
        text = re.sub(_GAP, "#", title or "", flags=re.I)
        if "#" not in text:
            return []
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß#\-]+", text)
        patterns = set()
        for i, tok in enumerate(tokens):
            if "#" not in tok:
                continue
            for lo in range(max(0, i - 2), i + 1):
                for hi in range(i + 1, min(len(tokens), i + 3) + 1):
                    literal = "".join(tokens[lo:hi])
                    pat = "".join(".*" if ch == "#" else re.escape(normalize(ch)) for ch in literal.replace("-", ""))
                    if len(re.sub(r"[#]", "", literal)) >= 5:
                        patterns.add(pat)
        return sorted(patterns)

    def gap_match(self, title):
        """Stationen, deren Name (auch früherer) zu einem lückenhaften Titel passt: {Stationsschlüssel}."""
        found = set()
        for pat in self._gap_patterns(title):
            for skey, keys in self.old_keys.items():
                if any(re.fullmatch(pat, k) for k in keys):
                    found.add(skey)
        return found

    def spelling_check(self, answer):
        """Gibt (neue Station, Hinweis) zurück oder (None, None), wenn nichts zu ändern ist."""
        station = _field(answer, "Station")
        if _is_none(station) or not _field(answer, "Match type").lower().startswith("direct"):
            return None, None
        pkey, _ = self.names.canonical(station)
        title = self._title_text(answer)
        if re.search(_GAP, title, re.I):  # Titel mit Lücken: Ähnlichkeit ist unzuverlässig, Muster verwenden
            hits = self.gap_match(title)
            if not hits or pkey in hits or len(hits) > 1:
                return None, None  # Modell passt zum Muster, oder nicht eindeutig: nichts ändern
            new = self.display[next(iter(hits))]
            return new, f"spelling check: the readable parts of the title fit only '{new}'"
        grams = self._ngrams(title)
        if not grams:
            return None, None
        pred_ratio = self._best(grams, self.old_keys.get(pkey, {pkey}))
        if pred_ratio >= KEEP_SIMILARITY:
            return None, None
        best_key, best_ratio = None, 0.0
        for skey, keys in self.old_keys.items():
            r = self._best(grams, keys)
            if r > best_ratio:
                best_key, best_ratio = skey, r
        if best_key and best_key != pkey and best_ratio >= MIN_SIMILARITY and best_ratio - pred_ratio >= MIN_GAP:
            new = self.display[best_key]
            return new, (f"spelling check: the title text is much more similar to '{new}' "
                         f"(similarity {best_ratio:.2f}) than to '{station}' ({pred_ratio:.2f})")
        return None, None

    # -- Linie ----------------------------------------------------------------------------------
    def line_check(self, answer):
        """Gibt einen Hinweis zurück, wenn die im Plan genannte Linie nicht zur Station passt."""
        station = _field(answer, "Station")
        if _is_none(station) or not _field(answer, "Match type").lower().startswith("nearby"):
            return None
        year = re.search(r"\b(1[89]\d\d|20\d\d)\b", _field(answer, "Plan date"))
        if not year or int(year.group(1)) < LINE_CHECK_FROM:
            return None  # ohne (ausreichend jungen) Plan-Datum ist die Liniennummer kein verlässlicher Hinweis
        text = _field(answer, "Title") + " " + _field(answer, "Recognized text")
        mentioned = {f"U{m}" for m in re.findall(r"(?<![A-Za-z])U[\s\-]?(\d{1,2})(?!\d)", text)}
        mentioned |= {"U5" for x in mentioned if x == "U55"}
        pkey, _ = self.names.canonical(station)
        have = self.lines.get(pkey, set())
        have = have | ({"U5"} if "U55" in have else set())
        if mentioned and have and not (mentioned & have):
            return (f"line check: the plan mentions {', '.join(sorted(mentioned))}, but {station} is served by "
                    f"{', '.join(sorted(have))} according to the station list")
        return None

    # -- Anwenden -------------------------------------------------------------------------------
    def apply(self, answer):
        """Wendet beide Prüfungen an; gibt (Antworttext, [Korrekturen]) zurück."""
        corrections, notes = [], []
        station = _field(answer, "Station")
        dnew, dinfo = self.date_rule(answer)
        new, info = (None, None) if dnew else self.spelling_check(answer)
        if dnew:
            answer = _replace_field(answer, "Station", dnew)
            answer = _replace_field(answer, "Confidence", ("high" if _year(_field(answer, "Plan date")) else "medium") + " – " + dinfo)
            corrections.append({"type": "date", "from": station, "to": dnew, "note": dinfo})
            notes.append(dinfo)
        elif new:
            answer = _replace_field(answer, "Station", new)
            answer = _replace_field(answer, "Confidence", "medium – " + info)
            corrections.append({"type": "spelling", "from": station, "to": new, "note": info})
            notes.append(info)
        else:
            note = self.line_check(answer)
            if note:
                if LINE_CHECK_ACTION == "none":
                    answer = _replace_field(answer, "Station", "none")
                    answer = _replace_field(answer, "Match type", "none (set by line check)")
                    corrections.append({"type": "line", "from": station, "to": "none", "note": note})
                else:  # Station bleibt, wird aber als unsicher markiert (Liniennummern auf Plänen können geplante
                    #     oder frühere Linien meinen)
                    corrections.append({"type": "line-flag", "from": station, "to": station, "note": note})
                answer = _replace_field(answer, "Confidence", "low – " + note)
                notes.append(note)
        if notes:
            answer = answer.rstrip() + "\n**Post-check:** " + "; ".join(notes) + "\n"
        return answer, corrections
