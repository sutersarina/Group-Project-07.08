# AI-GENERATED: this whole file was written by Claude (Anthropic)

"""
Collects the Flatfox rental listings of the canton of St. Gallen in an archive every day.
 
- Each listing appears only ONCE in archiv.csv (recognised by its id).
- "zuerst_gesehen" (first seen): day the listing was online for the first time.
- "zuletzt_gesehen" (last seen): last day the listing was still online.
- Listings that have not been online for more than 12 months are deleted.
 
The column names in archiv.csv stay in German on purpose: streamlit_app.py reads
them, and the existing archive already uses them.
 
Started automatically every day by GitHub Actions
(see .github/workflows/archiv_taeglich.yml). To start it by hand:
    python sammeln.py
"""
 
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
 
import pandas as pd
import requests
 
FOLDER = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(FOLDER, "archiv.csv")
CANTON_BORDER = os.path.join(FOLDER, "kanton_sg.geojson")
KEEP_DAYS = 365  # 12 months
 
FLATFOX_URL = "https://flatfox.ch/api/v1/public-listing/"
POSTCODE_URL = "https://openplzapi.org/ch/Cantons/17/Localities"  # 17 = canton of St. Gallen
PAGE_SIZE = 100   # maximum the Flatfox API returns per request
PARALLEL = 8      # this many pages are loaded at the same time
 
# These features become their own yes/no columns
FEATURES = [
    "balconygarden", "lift", "washingmachine", "tumbler", "dishwasher",
    "view", "parkingspace", "garage", "fireplace", "petsallowed",
]
 
 
def load_sg_postcodes():
    """Loads all localities in the canton of SG with their municipality and district.
 
    One postcode can belong to several localities (e.g. 9500 Wil),
    so there is a list per postcode.
    """
    postcodes = {}
    page = 1
    while True:
        response = requests.get(POSTCODE_URL, params={"page": page, "pageSize": 50}, timeout=30)
        response.raise_for_status()
        entries = response.json()
        if not entries:
            break
        for e in entries:
            postcodes.setdefault(int(e["postalCode"]), []).append({
                "ortschaft": e.get("name", ""),        # locality
                "gemeinde": e["commune"]["name"],      # municipality (column in archiv.csv)
                "wahlkreis": e["district"]["shortName"],  # district (column in archiv.csv)
            })
        page += 1
    return postcodes
 
 
def simplify(name):
    """'Gossau SG' -> 'gossau', 'Zuzwil (SG)' -> 'zuzwil', 'Sankt Gallen' -> 'st gallen'"""
    name = str(name or "").split(",")[0]               # 'Widnau, Sankt Gallen' -> 'Widnau'
    name = re.sub(r"\(.*?\)", "", name).lower().strip()
    name = re.sub(r"^(sankt\s+|st\.\s*|st\s+)", "st ", name)   # St.Gallen, St Gallen, Sankt Gallen
    return re.sub(r"\s+(sg|zh|tg|ar|ai|gr|gl|sz)$", "", name).strip()
 
 
def find_place(localities, city):
    """If a postcode has several localities, picks the matching one (by the place name)."""
    city = simplify(city)
    # Search by municipality first: the same locality (e.g. "Wil SG") can belong
    # to several municipalities (Wil, Kirchberg, Zuzwil ...)
    for loc in localities:
        if simplify(loc["gemeinde"]) == city:
            return loc
    for loc in localities:
        if simplify(loc["ortschaft"]) == city:
            return loc
    return localities[0]
 
 
def load_canton_border():
    """Reads the outline of the canton of SG (list of polygons; the first ring is
    the outer edge, further rings are holes, e.g. the cantons of Appenzell)."""
    with open(CANTON_BORDER, encoding="utf-8") as f:
        return json.load(f)["geometry"]["coordinates"]
 
 
def point_in_ring(lon, lat, ring):
    """Ray casting: counts how often a line from the point to the right crosses the edge."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
 
 
def is_in_canton(lon, lat, border):
    for polygon in border:
        edge, holes = polygon[0], polygon[1:]
        if point_in_ring(lon, lat, edge) and not any(point_in_ring(lon, lat, h) for h in holes):
            return True
    return False
 
 
def load_page(offset):
    """Loads one page with up to 100 listings."""
    response = requests.get(
        FLATFOX_URL, params={"limit": PAGE_SIZE, "offset": offset}, timeout=60
    )
    response.raise_for_status()
    return response.json()
 
 
def load_flatfox_listings():
    """Loads all listings in Switzerland (the API cannot filter by canton)."""
    first = load_page(0)
    offsets = range(PAGE_SIZE, first["count"], PAGE_SIZE)
    listings = list(first["results"])
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for data in pool.map(load_page, offsets):
            listings.extend(data["results"])
    return listings
 
 
def feature_names(attributes):
    """Features can come as a list of texts or as a list of {"name": ...}."""
    names = set()
    for a in attributes or []:
        names.add(a["name"] if isinstance(a, dict) else a)
    return names
 
 
def build_table(all_listings, sg_postcodes):
    """Turns the Flatfox data into a table, only rental apartments and houses in the canton of SG."""
    border = load_canton_border()
    rows = []
    for i in all_listings:
        if i.get("offer_type") != "RENT":
            continue
        if i.get("object_category") not in ("APARTMENT", "HOUSE"):
            continue
        if i.get("zipcode") is None or int(i["zipcode"]) not in sg_postcodes:
            continue
        # Some postcodes cross the canton border (e.g. 8630 Rüti ZH),
        # so we also check whether the apartment really lies in the canton of SG
        if i.get("latitude") and i.get("longitude"):
            if not is_in_canton(i["longitude"], i["latitude"], border):
                continue
 
        place = find_place(sg_postcodes[int(i["zipcode"])], i.get("city"))
        features = feature_names(i.get("attributes"))
        # The keys are the column names of archiv.csv (German, read by streamlit_app.py)
        row = {
            "id": i.get("pk"),
            "link": "https://flatfox.ch" + i["url"] if i.get("url") else None,
            "kategorie": i.get("object_category"),       # category
            "typ": i.get("object_type"),                 # type
            "plz": i.get("zipcode"),                     # postcode
            "ort": i.get("city"),                        # town
            "gemeinde": place["gemeinde"],               # municipality
            "wahlkreis": place["wahlkreis"],             # district
            "miete_netto": i.get("rent_net"),            # net rent
            "nebenkosten": i.get("rent_charges"),        # utilities
            "miete_brutto": i.get("rent_gross"),         # gross rent
            "zimmer": float(i["number_of_rooms"]) if i.get("number_of_rooms") else None,  # rooms
            "flaeche_m2": i.get("surface_living"),       # living area
            "baujahr": i.get("year_built"),              # year built
            "renovationsjahr": i.get("year_renovated"),  # year renovated
            "stockwerk": i.get("floor"),                 # floor
            "moebliert": i.get("is_furnished"),          # furnished
            "befristet": i.get("is_temporary"),          # temporary
            "lat": i.get("latitude"),
            "lon": i.get("longitude"),
        }
        for f in FEATURES:
            row[f] = f in features
        rows.append(row)
 
    df = pd.DataFrame(rows)
    if not df.empty:
        # Net rent per m², only where both values exist
        df["miete_pro_m2"] = (df["miete_netto"] / df["flaeche_m2"]).round(2)
    return df
 
 
def get_todays_listings():
    """All listings that are online in the canton of SG today."""
    return build_table(load_flatfox_listings(), load_sg_postcodes())
 
 
def update_archive(archive, today_df, today):
    """Adds today's listings to the archive and deletes old ones.
 
    - new listing: zuerst_gesehen = today
    - known listing: zuerst_gesehen stays, the other values are
      updated (e.g. if the rent was changed)
    - all of today's listings: zuletzt_gesehen = today
    """
    today_df = today_df.copy()
    today_df["zuletzt_gesehen"] = today
 
    if archive.empty:
        today_df["zuerst_gesehen"] = today
        new = today_df
    else:
        known = archive.set_index("id")["zuerst_gesehen"]
        today_df["zuerst_gesehen"] = today_df["id"].map(known).fillna(today)
        # Listings that are no longer online today stay unchanged in the archive
        no_longer_online = archive[~archive["id"].isin(today_df["id"])]
        new = pd.concat([no_longer_online, today_df], ignore_index=True)
 
    # 12-month limit: whatever has not been online for over a year is removed
    limit = (date.fromisoformat(today) - timedelta(days=KEEP_DAYS)).isoformat()
    new = new[new["zuletzt_gesehen"] >= limit]
 
    # Date columns first, newest listings at the top
    columns = ["id", "zuerst_gesehen", "zuletzt_gesehen"]
    new = new[columns + [c for c in new.columns if c not in columns]]
    return new.sort_values(["zuerst_gesehen", "id"], ascending=False)
 
 
def main():
    today = date.today().isoformat()
    archive = pd.read_csv(ARCHIVE) if os.path.exists(ARCHIVE) else pd.DataFrame()
    if not archive.empty:
        # Remove apartments outside the canton that got into the archive earlier
        border = load_canton_border()
        in_canton = [
            pd.isna(lat) or pd.isna(lon) or is_in_canton(lon, lat, border)
            for lat, lon in zip(archive["lat"], archive["lon"])
        ]
        archive = archive[in_canton]
    before = len(archive)
 
    print("Loading listings from Flatfox ...")
    today_df = get_todays_listings()
    if today_df.empty:
        # Better to save nothing than to overwrite the archive with an empty list
        raise SystemExit("No listings received, archive stays unchanged.")
 
    archive = update_archive(archive, today_df, today)
    archive.to_csv(ARCHIVE, index=False)
 
    new_count = (archive["zuerst_gesehen"] == today).sum()
    print(f"Online in the canton of SG today: {len(today_df)} listings, {new_count} of them new.")
    print(f"Archive: {before} listings before, {len(archive)} now.")
 
 
if __name__ == "__main__":
    main()
 