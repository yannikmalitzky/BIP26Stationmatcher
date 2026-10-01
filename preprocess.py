"""Preprocessing: Textregionen auf Bauplänen finden und in voller Auflösung ausschneiden.

Die Pläne sind oft 9000+ px breit, das Modell bekommt aber nur ein verkleinertes Bild, auf dem kleiner
Text unleserlich wird. Dieses Modul schickt deshalb stattdessen:
  1. eine Übersicht des ganzen Blatts (klein, für Layout und Kontext),
  2. den vermuteten Schriftfeld-Bereich in voller Auflösung,
  3. eine Collage der übrigen Textblöcke (Beschriftungen, Notizen) in voller Auflösung.

Erkennung (klassisch, mit OpenCV): Tinte finden -> lange Linien entfernen -> zeichengroße Komponenten
sammeln -> zu Textblöcken verschmelzen -> Schriftfeld = dichtester Textcluster nahe am Blattrand
(Ecke unten rechts bevorzugt).

Benötigt: pip install opencv-python-headless pillow numpy (pymupdf für PDF)
"""
import io
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None  # eigene Scans; das Pillow-Limit gegen "Decompression Bombs" ist hier unnötig

WORK_PX = 4000          # Erkennung läuft auf einer verkleinerten Kopie (längste Seite)
OVERVIEW_PX = 1600      # Übersichtsbild für das Modell
FALLBACK_PX = 3200      # ganzes Blatt, wenn keine Textregionen gefunden werden
MAX_CROP_PX = 2200      # längste Seite des Schriftfeld-Ausschnitts
COLLAGE_PX = 2400       # Kantenlänge der Collage mit den übrigen Textblöcken
MAX_OTHER = 14          # höchstens so viele weitere Textblöcke
CORNER = (0.68, 0.72, 1.0, 1.0)   # Zone unten rechts (übliches Schriftfeld): x0, y0, x1, y1 als Anteil des Blatts
CORNER_MAX_PX = 3000    # längste Seite des Eck-Ausschnitts (möglichst Originalauflösung)
ZONE_WORK_PX = 3000     # in der Eckzone wird (fast) in Originalauflösung erkannt
HUGE_PIXELS = 120e6     # darüber wird die Vollauflösung vorab halbiert (Speicher)


@dataclass
class Region:
    box: tuple          # (x0, y0, x1, y1) in Pixeln des Arbeitsbilds "gray"
    score: float = 0.0
    chars: int = 0


@dataclass
class Prepared:
    images: list = field(default_factory=list)   # [{"caption": str, "data": bytes, "mime": str}]
    overlay: bytes = b""                         # JPEG mit eingezeichneten Regionen (Debug)
    info: dict = field(default_factory=dict)
    # Textregionen für die manuelle Prüfung: [{"name": str, "box": [x0, y0, x1, y1]}] in Pixeln des geladenen
    # Graustufenbildes (load_gray); size = (Breite, Höhe) dieses Bildes
    regions: list = field(default_factory=list)
    size: tuple = (0, 0)


# ---------------------------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------------------------

def load_gray(data, filename, page=1):
    """Bytes (Bild oder PDF) -> Graustufenbild in voller Auflösung. page: Seite eines PDFs (ab 1)."""
    if filename.lower().endswith(".pdf"):
        try:
            import pymupdf
        except ImportError:
            raise RuntimeError("PDF support needs PyMuPDF: pip install pymupdf") from None
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            idx = min(max(int(page or 1), 1), pdf.page_count) - 1
            img = Image.open(io.BytesIO(pdf[idx].get_pixmap(dpi=200).tobytes("png")))
    else:
        img = Image.open(io.BytesIO(data))
    if img.mode != "L":
        img = img.convert("L")  # auch Modus "1" (Schwarz-Weiß-Scans)
    if img.width * img.height > HUGE_PIXELS:
        img = img.reduce(2)
    return img


# ---------------------------------------------------------------------------------------------
# Erkennung
# ---------------------------------------------------------------------------------------------

def _ink_mask(gray, s):
    """Binäre Tintenmaske im Arbeitsmaßstab s (Tinte = 255). Dünne Striche bleiben erhalten."""
    a = np.asarray(gray)
    _, ink = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if s < 1:
        size = (max(1, round(gray.width * s)), max(1, round(gray.height * s)))
        ink = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
        ink = (ink > 40).astype(np.uint8) * 255  # schon 16 % Tintenanteil zählt als Tinte
    return ink


def detect_text_blocks(gray, work_px=WORK_PX, cmin=5):
    """Findet Textblöcke; gibt (blöcke, rahmen, s, (w, h)) zurück. Alles in Pixeln des Arbeitsbilds."""
    s = min(1.0, work_px / max(gray.size))
    ink = _ink_mask(gray, s)
    h, w = ink.shape
    long_side = max(w, h)

    # 1) lange waagerechte/senkrechte Linien (Rahmen, Raster, Gleise) entfernen
    k = max(25, int(0.03 * long_side))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    lines = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    text_ink = cv2.bitwise_and(ink, cv2.bitwise_not(lines))

    # 2) zeichengroße Komponenten
    n, _, stats, cents = cv2.connectedComponentsWithStats(text_ink, connectivity=8)
    cmax = max(30, int(0.02 * long_side))
    chars = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if not (cmin <= max(cw, ch) <= cmax) or min(cw, ch) < 2 or area < 6:
            continue
        fill = area / float(cw * ch)
        if fill < 0.08 or not (0.08 <= cw / ch <= 12):
            continue
        chars.append((x, y, cw, ch, cents[i]))
    if len(chars) < 8:
        return [], [], s, (w, h)

    # 3) Zeichen zu Textblöcken verschmelzen
    med = float(np.median([max(c[2], c[3]) for c in chars]))
    dk = int(min(45, max(7, 1.7 * med)))
    mask = np.zeros((h, w), np.uint8)
    for x, y, cw, ch, _ in chars:
        mask[y:y + ch, x:x + cw] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)))
    n2, lab, st2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    counts = np.zeros(n2, int)
    for x, y, cw, ch, (cx, cy) in chars:
        counts[lab[min(h - 1, int(cy)), min(w - 1, int(cx))]] += 1

    # Zeichen je Block sammeln, um Schraffuren auszusortieren: dort sind die "Zeichen" fast immer
    # lange, dünne Linienstücke und die Fläche ist dicht mit Tinte gefüllt
    per_block = {}
    for x, y, cw, ch, (cx, cy) in chars:
        per_block.setdefault(lab[min(h - 1, int(cy)), min(w - 1, int(cx))], []).append((cw, ch))

    integral = cv2.integral((ink > 0).astype(np.uint8))

    def ink_share(x0, y0, x1, y1):  # Tintenanteil in einem Rechteck (mit Linien)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        area = max(1, (x1 - x0) * (y1 - y0))
        return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area

    blocks = []
    for i in range(1, n2):
        x, y, bw, bh, _ = st2[i]
        if counts[i] < 5 or bw * bh > 0.45 * w * h:
            continue
        m = int(0.6 * max(bw, bh))  # Umgebung: in Schraffuren ist auch die Nachbarschaft voller Tinte
        if ink_share(x - m, y - m, x + bw + m, y + bh + m) > 0.28:
            continue
        if counts[i] / float(bw * bh) < 2e-4:
            continue
        comps = per_block.get(i, [])
        elongated = sum(1 for cw, ch in comps if max(cw, ch) / max(1, min(cw, ch)) > 4) / max(1, len(comps))
        ink_ratio = float(np.count_nonzero(text_ink[y:y + bh, x:x + bw])) / (bw * bh)
        if elongated > 0.5 or ink_ratio > 0.38:
            continue
        blocks.append(Region((int(x), int(y), int(x + bw), int(y + bh)), chars=int(counts[i])))

    # Rahmen: zusammenhängende Linienflächen (z. B. der Rahmen des Schriftfelds); zu große (Blattrand, Raster) fallen weg
    nl, _, sl, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    frames = [(int(x), int(y), int(x + fw), int(y + fh)) for x, y, fw, fh, _ in sl[1:]
              if fw > 0.04 * w and fh > 0.02 * h and fw < 0.45 * w and fh < 0.45 * h]
    return blocks, frames, s, (w, h)


def _edge_score(box, w, h):
    """Bonus für Bereiche nahe am Blattrand, besonders in der Ecke unten rechts (übliches Schriftfeld)."""
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    bonus = 1.0 + (1.0 if min(cx, 1 - cx, cy, 1 - cy) < 0.22 else 0.0)
    if cx > 0.5 and cy > 0.6:
        bonus *= 2.5
    return bonus


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def find_title_block(blocks, frames, w, h):
    """Schriftfeld = Textcluster mit der höchsten randgewichteten Zeichenzahl; rastet auf einen Rahmen ein."""
    if not blocks:
        return None, blocks
    best, best_score = None, -1.0
    for seed in blocks:  # Cluster = alle Blöcke im Umkreis eines Blocks
        x0, y0, x1, y1 = seed.box
        win = (x0 - 0.15 * w, y0 - 0.08 * h, x1 + 0.15 * w, y1 + 0.08 * h)
        members = [b for b in blocks if b.box[0] >= win[0] and b.box[1] >= win[1]
                   and b.box[2] <= win[2] and b.box[3] <= win[3]]
        box = _union([m.box for m in members])
        if (box[2] - box[0]) > 0.40 * w or (box[3] - box[1]) > 0.40 * h:
            members, box = [seed], seed.box  # zu groß: nur der Block selbst
        score = sum(m.chars for m in members) * _edge_score(box, w, h)
        if score > best_score:
            best, best_score, best_members, best_box = seed, score, members, box
    # auf den kleinsten umschließenden Rahmen einrasten (enthält auch Zellen mit wenig Text)
    inside = [f for f in frames if f[0] <= best_box[0] + 5 and f[1] <= best_box[1] + 5
              and f[2] >= best_box[2] - 5 and f[3] >= best_box[3] - 5]
    if inside:
        best_box = min(inside, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
    pad = int(0.008 * max(w, h))
    box = (max(0, best_box[0] - pad), max(0, best_box[1] - pad), min(w, best_box[2] + pad), min(h, best_box[3] + pad))
    title = Region(box, best_score, sum(m.chars for m in best_members))
    rest = [b for b in blocks if not (box[0] <= b.box[0] and box[1] <= b.box[1] and b.box[2] <= box[2] and b.box[3] <= box[3])]
    return title, rest


# ---------------------------------------------------------------------------------------------
# Bilder für das Modell
# ---------------------------------------------------------------------------------------------

def _jpeg(img, q=90):
    out = io.BytesIO()
    img.convert("L" if img.mode == "L" else "RGB").save(out, format="JPEG", quality=q)
    return out.getvalue()


def _fit(img, max_px):
    if max(img.size) > max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    return img


def _where(box, w, h):
    """Lage eines Bereichs auf dem Blatt in Prozent, z. B. 'x 62-98 %, y 80-99 %'."""
    p = lambda v, t: int(round(100 * v / t))
    return f"x {p(box[0], w)}-{p(box[2], w)} %, y {p(box[1], h)}-{p(box[3], h)} %"


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _collage(crops):
    """Packt (label, bild)-Paare zeilenweise auf eine Leinwand; gibt (bild, Anzahl gepackter) zurück."""
    W, gap, band = COLLAGE_PX, 10, 30
    font = _font(22)
    canvas = Image.new("L", (W, COLLAGE_PX), 255)
    draw = ImageDraw.Draw(canvas)
    x = y = row_h = 0
    packed = 0
    for label, img in crops:
        img = _fit(img, W - 2 * gap)
        cw, ch = img.size
        if x + cw > W - gap:  # neue Zeile
            x, y, row_h = 0, y + row_h + gap, 0
        if y + band + ch > COLLAGE_PX:
            break
        draw.text((x + 2, y), label, fill=0, font=font)
        canvas.paste(img, (x, y + band))
        draw.rectangle([x - 1, y + band - 1, x + cw, y + band + ch], outline=160)
        x += max(cw, 120) + gap
        row_h = max(row_h, band + ch)
        packed += 1
    used_h = min(COLLAGE_PX, y + row_h + gap)
    return canvas.crop((0, 0, W, max(used_h, 60))), packed


def _scaled(box, f, dx=0, dy=0):
    return (int(round(box[0] * f)) + dx, int(round(box[1] * f)) + dy,
            int(round(box[2] * f)) + dx, int(round(box[3] * f)) + dy)


def _inside(box, zone):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def _overlap_share(a, b):
    """Anteil der Fläche von b, der von a bedeckt wird."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return (w * h) / float((b[2] - b[0]) * (b[3] - b[1])) if w > 0 and h > 0 else 0.0


def prepare(data, filename, page=1):
    """Bytes eines Plans -> Prepared (Bilder für das Modell, Debug-Overlay, Infos)."""
    gray = load_gray(data, filename, page)
    W, H = gray.size

    # 1) Erkennung über das ganze Blatt auf verkleinerter Kopie -> Koordinaten in Pixeln von "gray"
    blocks, frames, s, _ = detect_text_blocks(gray)
    blocks = [Region(_scaled(b.box, 1 / s), chars=b.chars) for b in blocks]
    frames = [_scaled(f, 1 / s) for f in frames]

    # 2) Eckzone unten rechts noch einmal in (fast) Originalauflösung: findet auch winzigen Text
    zone = tuple(int(v) for v in (CORNER[0] * W, CORNER[1] * H, CORNER[2] * W, CORNER[3] * H))
    zb, _, zs, _ = detect_text_blocks(gray.crop(zone), work_px=ZONE_WORK_PX, cmin=3)
    zone_blocks = [Region(_scaled(b.box, 1 / zs, zone[0], zone[1]), chars=b.chars) for b in zb]
    if zone_blocks:  # Ergebnisse der groben Erkennung in dieser Zone werden durch die feinere ersetzt
        blocks = [b for b in blocks if not _inside(b.box, zone)] + zone_blocks

    title, rest = find_title_block(blocks, frames, W, H)

    overview = _fit(gray, OVERVIEW_PX)
    prep = Prepared()

    if title is None:  # nichts gefunden: wie bisher das ganze Blatt größer schicken
        big = _fit(gray, FALLBACK_PX)
        prep.images.append({"caption": "Image 1: the whole sheet (no separate text regions could be detected).",
                            "data": _jpeg(big), "mime": "image/jpeg"})
        prep.overlay = _jpeg(overview.convert("RGB"))
        prep.info = {"title_block": False, "regions": 0, "images": 1, "size": [W, H]}
        prep.size = (W, H)
        prep.regions = [{"name": "Corner (bottom right)", "box": list(zone)}]
        return prep

    # Übersicht
    prep.images.append({
        "caption": "Image 1: overview of the whole sheet (reduced resolution, for layout and context).",
        "data": _jpeg(overview), "mime": "image/jpeg"})

    # Schriftfeld in voller Auflösung
    tb_img = _fit(gray.crop(title.box), MAX_CROP_PX)
    prep.images.append({
        "caption": ("Image 2: the most likely title block, cut out of the sheet at full resolution; located at "
                    + _where(title.box, W, H) + " of the sheet."),
        "data": _jpeg(tb_img), "mime": "image/jpeg"})

    # Ecke unten rechts immer in voller Auflösung mit (dort sitzt meist das Schriftfeld, auch wenn die
    # Erkennung den Text nicht sieht). Entfällt, wenn das Schriftfeld die Ecke ohnehin abdeckt.
    corner_box = zone
    corner_sent = _overlap_share(title.box, corner_box) < 0.6
    if corner_sent:
        prep.images.append({
            "caption": ("Image 3: the bottom right corner of the sheet at full resolution (" + _where(corner_box, W, H)
                        + "); the title block is usually here, even small text can be important."),
            "data": _jpeg(_fit(gray.crop(corner_box), CORNER_MAX_PX)), "mime": "image/jpeg"})

    # übrige Textblöcke (nach Zeichenzahl, mit Rand für Kontext)
    rest = [b for b in rest if not (corner_sent and _overlap_share(corner_box, b.box) > 0.7)]  # schon im Eck-Ausschnitt
    rest = sorted(rest, key=lambda b: -b.chars)[:MAX_OTHER]
    crops, order = [], []
    pad = int(0.004 * max(W, H))
    for i, b in enumerate(rest, 1):
        box = (max(0, b.box[0] - pad), max(0, b.box[1] - pad), min(W, b.box[2] + pad), min(H, b.box[3] + pad))
        crops.append((f"#{i}: {_where(box, W, H)}", gray.crop(box)))
        order.append((i, box))
    packed = 0
    if crops:
        canvas, packed = _collage(crops)
        prep.images.append({
            "caption": (f"Image {len(prep.images) + 1}: further text regions of the sheet at full resolution, each with "
                        "a label that gives its position on the sheet."),
            "data": _jpeg(canvas), "mime": "image/jpeg"})

    # Debug-Overlay auf der Übersicht: rot = Schriftfeld, gelb = Eck-Ausschnitt, blau = weitere Textblöcke
    ov = overview.convert("RGB")
    d = ImageDraw.Draw(ov)
    f = ov.width / W
    if corner_sent:
        d.rectangle([v * f for v in corner_box], outline=(230, 170, 0), width=3)
    d.rectangle([v * f for v in title.box], outline=(220, 0, 0), width=4)
    for i, box in order:
        d.rectangle([v * f for v in box], outline=(0, 90, 220) if i <= packed else (150, 150, 150), width=2)
    prep.overlay = _jpeg(ov)
    prep.size = (W, H)
    prep.regions = [{"name": "Title block", "box": list(title.box)},
                    {"name": "Corner (bottom right)", "box": list(corner_box)}]
    prep.regions += [{"name": f"Text region #{i}", "box": list(box)} for i, box in order]
    prep.info = {"title_block": True, "regions": packed, "found": len(order), "images": len(prep.images),
                 "corner_crop": corner_sent, "corner_blocks": len(zone_blocks),
                 "size": [W, H], "title_block_box": _where(title.box, W, H)}
    return prep
