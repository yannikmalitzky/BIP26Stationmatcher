"""Bahnhofsliste von berliner-u-bahn.info einlesen (HTML-Tabelle -> Python-Daten).

Benötigt: pip install requests beautifulsoup4
Hinweis: Die Seite ist urheberrechtlich geschützt (© Patrick Popiol).
Für eigene Auswertungen ok, für Veröffentlichung bitte vorher beim Betreiber anfragen.
"""
import csv
import re
import requests
from bs4 import BeautifulSoup

URL = "http://berliner-u-bahn.info/bhfdaten.php"
HEADERS = {"User-Agent": "bhf-scraper/1.0 (yannik@malitzky.de)"}


def _text(zelle):
    """Zelleninhalt als Text, Zeilenumbrüche (<br>) als '; '."""
    teile = [t.strip() for t in zelle.get_text("\n").split("\n")]
    return "; ".join(t for t in teile if t)


def parse_bahnhoefe(html):
    soup = BeautifulSoup(html, "html.parser")
    tabelle = soup.find("table")
    bahnhoefe = []

    for tr in tabelle.find_all("tr"):
        zellen = tr.find_all("td")
        if len(zellen) != 5:
            continue  # Kopfzeile / leere Zeilen

        name_zelle, kuerzel, linie, eroeffnung, bemerkung = zellen

        # Das "*" (Link auf die Fußnote) markiert nicht mehr existierende Bahnhöfe
        fussnote = name_zelle.find("a", attrs={"data-src": "#footnote"})
        if fussnote:
            fussnote.extract()
        name = " ".join(name_zelle.get_text(" ").split())

        # Linien stecken als versteckter Text in <span class="visuallyhidden">
        versteckt = " ".join(s.get_text() for s in linie.select(".visuallyhidden"))
        linien = list(dict.fromkeys(re.findall(r"U\d+", versteckt or linie.get_text())))

        bem = _text(bemerkung)
        heute = re.search(r"heute:[\s;]*(.+?)(;|$)", bem, re.IGNORECASE)

        bahnhoefe.append({
            "name": name,
            "kuerzel": _text(kuerzel) or None,
            "linien": linien,
            "eroeffnung": _text(eroeffnung) or None,
            "bemerkung": bem or None,
            "existiert": fussnote is None,
            # Zeilen mit "heute: X" sind frühere Namen, die auf einen aktuellen Bahnhof verweisen
            "heute": heute.group(1).strip() if heute else None,
        })
    return bahnhoefe


def lade_bahnhoefe():
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return parse_bahnhoefe(resp.text)


def als_csv_speichern(bahnhoefe, pfad="bhfdaten.csv"):
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bahnhoefe[0].keys())
        writer.writeheader()
        for b in bahnhoefe:
            writer.writerow({**b, "linien": ",".join(b["linien"])})


if __name__ == "__main__":
    bahnhoefe = lade_bahnhoefe()
    aktuell = [b for b in bahnhoefe if b["existiert"] and not b["heute"]]
    alte_namen = [b for b in bahnhoefe if b["heute"]]
    weg = [b for b in bahnhoefe if not b["existiert"]]
    print(f"{len(bahnhoefe)} Einträge: {len(aktuell)} Bahnhöfe, "
          f"{len(alte_namen)} frühere Namen, {len(weg)} nicht mehr existierend")
    als_csv_speichern(bahnhoefe)
    print("Gespeichert in bhfdaten.csv")