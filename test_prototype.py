import os

import config  # loads .env (HTW_API_KEY) into the environment

from openai import OpenAI
import base64
import csv
import mimetypes
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

API_KEY = os.environ.get("HTW_API_KEY")  # never commit the key: export HTW_API_KEY=...
if not API_KEY:
    raise SystemExit("Set the environment variable HTW_API_KEY (see .env.example).")

BASE_URL = os.environ.get("HTW_BASE_URL", "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1")

# Change this to a vision-capable model available on the server
MODEL = "qwen3.8-27b"

STATIONS_CSV = Path(__file__).parent / "bhfdaten.csv"

# Columns passed to the model
CSV_COLUMNS = [
    "name",
    "kuerzel",
    "linien",
    "eroeffnung",
    "bemerkung",
    "existiert",
    "heute",
]

# Plans are often 9000+ px wide; the model does not need that much. Downscaling the
# image before sending cuts upload and prompt size; the web UI reports when it happens.
MAX_IMAGE_PX = 3200

# Sampling temperature: 0 makes runs reproducible, so prompt changes can be compared fairly.
TEMPERATURE = 0

# The model is a reasoning model: with thinking enabled an analysis takes ~90-130 s,
# without it ~10 s (measured with a 9440x6720 plan).
THINKING = False

# The input is comma-separated, but "linien" and "bemerkung" contain commas and
# semicolons, so the text sent to the model uses "|" as separator.
OUT_SEP = "|"


# ============================================================
# CONNECT TO HTW API
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# OPTIONALLY LOAD STATION LIST
# ============================================================

def load_stations_text(path):
    """Return the station CSV reduced to CSV_COLUMNS as plain text."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [OUT_SEP.join(CSV_COLUMNS)]
    for row in rows:
        lines.append(OUT_SEP.join((row.get(col) or "").replace(OUT_SEP, "/") for col in CSV_COLUMNS))
    return "\n".join(lines), len(rows)


# ============================================================
# BUILD PROMPT
# ============================================================

PROMPT_TASK = """\
You are analyzing one document from a folder of construction plans (technical drawings) \
from Berlin's U-Bahn network. Your task is to map this document to the Berlin U-Bahn \
station it belongs to, if it belongs to one. Proceed in exactly these steps:

1. Is it a standalone facility? Decide first what the plan depicts. Standalone \
facilities belong to NO station. These are facilities with their own purpose that are \
not part of the line or a station, for example: rectifier substations \
(Gleichrichterwerk, Unterwerk), depots and workshops (Betriebswerkstatt, Kehranlage), \
and self-contained tunnel structures of their own such as a tunnel under a river or \
canal. If the plan clearly and explicitly shows such a standalone facility (for example \
the title names it, or the drawing itself unmistakably shows it), the answer is "none" \
and you are done. A missing location is NOT evidence for a standalone facility: a plan \
that shows a detail (for example railings, stairs, reinforcement, a wall or a section) \
without a place name still belongs to some station, and an experienced person can often \
tell which one from the drawing itself. Do not classify a plan as a generic or standard \
drawing just because no street or station is named; treat it as a plan of the U-Bahn \
network and continue.
Everything else in the U-Bahn network is in principle assigned to a station, even if it \
is not the station building itself: station halls, platforms, entrances, stairs, \
elevators and technical rooms, but also bridges, tunnel sections, open track sections, \
ventilation and emergency-exit shafts, excavation pits and other line structures. They \
belong to the station they are located at or closest to (for example an overpass at a \
street belongs to the station named after that street or nearest to it). Do NOT answer \
"none" merely because a plan shows a bridge, tunnel or line structure: continue with \
the next steps and find the location. The same applies to buildings next to the line \
that are affected by the U-Bahn construction (for example underpinning or shoring of a \
neighboring house): they are not standalone facilities and are assigned by location like \
any other line structure. If the subject is unclear, continue as well. In the next steps \
use not only the text but also the content of the drawing (layout of rooms, platforms, \
stairs, tracks, directions, neighboring buildings, street layout) to work out where the \
plan belongs.

2. Extract the text. Read all text visible in the plan and note WHERE on the sheet it \
stands: in the title block, in the plan header, in the drawing area (labels, street \
names, room names, dimensions, notes), or in a margin, legend or stamp. Also find the \
DATE of the plan (date field in the title block, revision table, stamps, dates in \
notes); it is needed to interpret station names correctly in step 4.

3. Decide what the title of the plan is, based on the position and role of the text. \
The title block is often a bordered box in the bottom right corner of the sheet, but \
not always: it can also be at the bottom, at the top, or in another corner. It \
contains the title of the drawing, the project or building name, the location and the \
drawing number. Text in the drawing area, such as street names, names of neighboring \
buildings or dimensions, is orientation and is NOT the title: a street called \
"Tierpark" printed in the drawing does not mean the plan belongs to station Tierpark. \
Never take a street name as the title just because it is printed prominently.

4. Direct match: if the title (or the plan header) names a U-Bahn station (e.g. \
"Haltestelle X", "Bahnhof X", "U-Bhf. X", "Station X", or a former name of a \
station), match it against the stations. Take current AND former station names into \
account, and match on spelling variants (e.g. "Str." / "Straße", with or without the \
prefix "U-Bahnhof"). The text of the plan may be misread or misspelled (old lettering, \
look-alike letters such as rn/m, I/l/1 or h/b, a wrongly read first syllable). If a name \
in the title does not match any station exactly, look in the list for the entry with the \
most similar letters and accept it when the similarity is clear, before you try any other \
reasoning. The name may also be damaged: letters missing, a line struck through it, or \
parts covered. Use the readable parts (beginning, end, letters in between) and match them \
against the list; accept a station when exactly one list entry fits the readable parts, or \
one fits clearly best. Never use former names from your own knowledge: a former name \
counts only if it appears in the list (columns name and bemerkung). Station names change over time and the same name can belong to \
different stations at different times: a station may have been renamed, and a NEW \
station may later have been built and given the old name (for example an old station \
renamed in 1929 and a new station of the same old name opened in 1978). Use the date of \
the plan together with the opening dates and the name history in the list (columns \
eroeffnung and bemerkung) to decide which station carried that name when the plan was \
drawn. This date rule is binding: a plan drawn after a new station opened under that name \
belongs to the new station; a plan drawn before the rename belongs to the station that \
had the name then. If the plan has no date, take the station that carries the name today.

5. Nearby match: if the title does not name a station (this includes a title whose name is \
too damaged to match, fits no entry of the list, or does not appear in the list at all), \
use ALL location information \
in the plan (street names, addresses, cross streets, squares, the street a bridge \
crosses, neighboring buildings) to locate the plan in Berlin (district, position) and \
find the nearest U-Bahn station it belongs to. For bridges, tunnels and other line \
structures, the street or area they are named after or cross is a strong location hint. \
Estimate the walking distance of each candidate and consider whether the street runs \
past an entrance or is only a cross street. Choose the most plausible station and name \
the runner-up stations you rejected and why. Do not answer "none" merely because the name \
in the title could not be matched: try this step first. Check the candidate against the list: if the \
plan names a line that you can identify with a line in the list (for example "U5"), the \
station must be served by that line (column linien); reject candidates that are not. \
Rivers, canals, river banks and names of tunnels or bridges alone are not location \
evidence for a station; use street names, squares, addresses and named station \
structures that you can connect to a specific station. If the only clue is such a \
feature, the answer is "none".

6. If neither the title nor the location information lets you assign a station with \
reasonable plausibility (for example a tunnel or bridge plan without any indication of \
where it is), the answer is "none".

Confidence: "high" if a station is named in the title block or plan header, "medium" \
for a well-founded nearby match from street names or addresses, "low" for weak evidence.
"""

PROMPT_STATION_LIST = """\

Use the following list of Berlin U-Bahn stations as the set of possible answers for \
steps 4 and 5. \
It is a "|"-separated table with these columns:
- name: station name as listed (may be a former name, a variant, or carry a platform \
suffix such as "[oben]" / "[Bstg.A]")
- kuerzel: station abbreviation
- linien: lines serving the station (comma-separated; empty for name-only entries)
- eroeffnung: opening date (dd.mm.yyyy)
- bemerkung: remarks such as former names with dates, closures and reopenings \
("heute:" = this entry is an old name, see column heute)
- existiert: True = entry exists / station is in operation, False = closed
- heute: if the entry is a former name or variant, the current station name; use the \
row with this name to get lines and status

The Station you name MUST be an entry of this list. If the matching entry is a former \
name or variant (column heute is filled), name the CURRENT station from column heute \
instead, unless the plan date shows that the name referred to a different station at the \
time (see step 4): if the same name is used by two entries, one an old name (heute filled) \
and one a station that opened later under that name, pick by the plan date. Use your own knowledge of Berlin \
geography only to locate a street or address and to judge distances to stations. Take \
lines, opening/closing dates and renamings ONLY from the list; if a piece of information \
is missing there, write "not in the list" instead of using your own knowledge.

{stations}
"""

PROMPT_NO_LIST = """\

No station list is provided. Use your own knowledge of current and historical Berlin \
U-Bahn stations and mark clearly in the answer that the information is not verified \
against a list.
"""

PROMPT_OUTPUT = """\

Answer in English using exactly this structure and keep every field short. Do not use \
bullet points or extra formatting:

**Belongs to a station:** <yes / no / unclear> – <what the plan depicts; "no" only for a standalone facility such as a rectifier substation>
**Recognized text:** <relevant text fragments with their position on the sheet>
**Plan date:** <date or year found on the plan and where; "not found" if none>
**Title:** <the text you consider the title of the plan and where it stands; "not identifiable" if none>
**Match type:** <direct (station named in the title) / nearby (derived from street names or addresses) / none>
**Nearby stations:** <for a nearby match: candidate stations with approximate walking distance; otherwise "n/a">
**Station:** <name of the station the plan belongs to, exactly as in the list, or "none">
**Additional stations:** <only if the title or the subject explicitly names more than one station (for example a section between two stations): the other stations exactly as in the list, comma-separated; otherwise "none">
**Facility type:** <only if Station is "none": what the plan shows, 1-3 words (for example rectifier substation, workshop, river tunnel; use "unclear location" if it belongs to a station that cannot be determined); otherwise "n/a">
**Confidence:** <high / medium / low> – <one short reason>
"""

PROMPT_IMAGES = """\

You receive several images of the same plan, each introduced by a caption: an overview \
of the whole sheet in reduced resolution, and cut-outs of regions that contain text at \
full resolution (the most likely title block and further text regions, each labelled \
with its position on the sheet as percentages of the sheet width and height). Read small \
text from the cut-outs, and use the overview for the layout and for what the plan \
depicts. Positions in the captions and labels tell you where the text stands on the \
sheet: the title block is usually at the bottom right, but the cut-out marked as \
"most likely title block" is only a guess. Judge yourself which text is the title.
"""


def build_prompt(stations_text=None, multi_image=False):
    prompt = PROMPT_TASK
    if multi_image:
        prompt += PROMPT_IMAGES
    prompt += PROMPT_STATION_LIST.format(stations=stations_text) if stations_text else PROMPT_NO_LIST
    return prompt + PROMPT_OUTPUT


# ============================================================
# SEND IMAGE TO MODEL
# ============================================================

def analyze_plan(image_bytes, mime_type, stations_text=None, thinking=THINKING):
    """Send the plan image to the model and return its answer as text."""
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(stations_text)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def analyze_images(images, stations_text=None, thinking=THINKING, timeout=None):
    """Send several images of one plan (overview + full-resolution text regions).

    images: [{"caption": str, "data": bytes, "mime": str}]
    timeout: seconds to wait for the answer (raises openai.APITimeoutError, no retries); None = no limit
    """
    content = [{"type": "text", "text": build_prompt(stations_text, multi_image=len(images) > 1)}]
    for img in images:
        b64 = base64.b64encode(img["data"]).decode("utf-8")
        content.append({"type": "text", "text": img["caption"]})
        content.append({"type": "image_url", "image_url": {"url": f"data:{img['mime']};base64,{b64}"}})
    api = client.with_options(timeout=timeout, max_retries=0) if timeout else client
    response = api.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":
    stations_text = None
    answer = input("Include the U-Bahn station list (CSV) in the request? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes", "j", "ja"):
        if STATIONS_CSV.exists():
            stations_text, count = load_stations_text(STATIONS_CSV)
            print(f"Loaded {count} stations from {STATIONS_CSV.name}\n")
        else:
            print(f"File not found: {STATIONS_CSV}. Continuing without it.\n")
    else:
        print("Continuing without the station list.\n")

    image_path = input("Enter the path to your image: ").strip()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    print("\nAnalyzing plan...\n")
    result = analyze_plan(image_bytes, mime_type, stations_text)

    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(result)
