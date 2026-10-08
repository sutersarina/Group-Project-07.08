"""
Sammelt jeden Tag die Flatfox-Mietinserate im Kanton St. Gallen in einem Archiv.

- Jedes Inserat steht nur EINMAL in archiv.csv (erkannt an seiner id).
- "zuerst_gesehen": Tag, an dem das Inserat zum ersten Mal online war.
- "zuletzt_gesehen": letzter Tag, an dem es noch online war.
- Inserate, die seit mehr als 12 Monaten nicht mehr online waren, werden gelöscht.

Wird täglich automatisch von GitHub Actions gestartet
(siehe .github/workflows/archiv_taeglich.yml). Von Hand starten:
    python sammeln.py
"""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

ARCHIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archiv.csv")
AUFBEWAHREN_TAGE = 365  # 12 Monate

FLATFOX_URL = "https://flatfox.ch/api/v1/public-listing/"
PLZ_URL = "https://openplzapi.org/ch/Cantons/17/Localities"  # 17 = Kanton St. Gallen
SEITENGROESSE = 100  # Maximum, das die Flatfox-API pro Anfrage liefert
GLEICHZEITIG = 8     # so viele Seiten werden parallel geladen

# Diese Ausstattungsmerkmale werden zu eigenen Ja/Nein-Spalten
MERKMALE = [
    "balconygarden", "lift", "washingmachine", "tumbler", "dishwasher",
    "view", "parkingspace", "garage", "fireplace", "petsallowed",
]


def lade_sg_postleitzahlen():
    """Holt alle Postleitzahlen im Kanton SG mit Gemeinde und Wahlkreis."""
    plz = {}
    seite = 1
    while True:
        antwort = requests.get(PLZ_URL, params={"page": seite, "pageSize": 50}, timeout=30)
        antwort.raise_for_status()
        eintraege = antwort.json()
        if not eintraege:
            break
        for e in eintraege:
            plz[int(e["postalCode"])] = {
                "gemeinde": e["commune"]["name"],
                "wahlkreis": e["district"]["shortName"],
            }
        seite += 1
    return plz


def lade_seite(offset):
    """Holt eine Seite mit bis zu 100 Inseraten."""
    antwort = requests.get(
        FLATFOX_URL, params={"limit": SEITENGROESSE, "offset": offset}, timeout=60
    )
    antwort.raise_for_status()
    return antwort.json()


def lade_flatfox_inserate():
    """Holt alle Inserate der Schweiz (die API kann nicht nach Kanton filtern)."""
    erste = lade_seite(0)
    offsets = range(SEITENGROESSE, erste["count"], SEITENGROESSE)
    inserate = list(erste["results"])
    with ThreadPoolExecutor(max_workers=GLEICHZEITIG) as pool:
        for daten in pool.map(lade_seite, offsets):
            inserate.extend(daten["results"])
    return inserate


def merkmal_namen(attributes):
    """Ausstattung kann als Liste von Texten oder von {"name": ...} kommen."""
    namen = set()
    for a in attributes or []:
        namen.add(a["name"] if isinstance(a, dict) else a)
    return namen


def baue_tabelle(alle, sg_plz):
    """Macht aus den Flatfox-Daten eine Tabelle, nur Mietwohnungen und Häuser im Kanton SG."""
    zeilen = []
    for i in alle:
        if i.get("offer_type") != "RENT":
            continue
        if i.get("object_category") not in ("APARTMENT", "HOUSE"):
            continue
        if i.get("zipcode") is None or int(i["zipcode"]) not in sg_plz:
            continue

        ort = sg_plz[int(i["zipcode"])]
        ausstattung = merkmal_namen(i.get("attributes"))
        zeile = {
            "id": i.get("pk"),
            "link": "https://flatfox.ch" + i["url"] if i.get("url") else None,
            "kategorie": i.get("object_category"),
            "typ": i.get("object_type"),
            "plz": i.get("zipcode"),
            "ort": i.get("city"),
            "gemeinde": ort["gemeinde"],
            "wahlkreis": ort["wahlkreis"],
            "miete_netto": i.get("rent_net"),
            "nebenkosten": i.get("rent_charges"),
            "miete_brutto": i.get("rent_gross"),
            "zimmer": float(i["number_of_rooms"]) if i.get("number_of_rooms") else None,
            "flaeche_m2": i.get("surface_living"),
            "baujahr": i.get("year_built"),
            "renovationsjahr": i.get("year_renovated"),
            "stockwerk": i.get("floor"),
            "moebliert": i.get("is_furnished"),
            "befristet": i.get("is_temporary"),
            "lat": i.get("latitude"),
            "lon": i.get("longitude"),
        }
        for m in MERKMALE:
            zeile[m] = m in ausstattung
        zeilen.append(zeile)

    df = pd.DataFrame(zeilen)
    if not df.empty:
        # Miete pro m² (netto), nur wo beides vorhanden ist
        df["miete_pro_m2"] = (df["miete_netto"] / df["flaeche_m2"]).round(2)
    return df


def hole_heutige_inserate():
    """Alle Inserate, die heute im Kanton SG online sind."""
    return baue_tabelle(lade_flatfox_inserate(), lade_sg_postleitzahlen())


def aktualisiere_archiv(archiv, heute_df, heute):
    """Fügt die heutigen Inserate ins Archiv ein und löscht alte.

    - neues Inserat: zuerst_gesehen = heute
    - bekanntes Inserat: zuerst_gesehen bleibt, die übrigen Werte werden
      aktualisiert (z.B. wenn die Miete geändert wurde)
    - alle heutigen Inserate: zuletzt_gesehen = heute
    """
    heute_df = heute_df.copy()
    heute_df["zuletzt_gesehen"] = heute

    if archiv.empty:
        heute_df["zuerst_gesehen"] = heute
        neu = heute_df
    else:
        bekannt = archiv.set_index("id")["zuerst_gesehen"]
        heute_df["zuerst_gesehen"] = heute_df["id"].map(bekannt).fillna(heute)
        # Inserate, die heute nicht mehr online sind, bleiben unverändert im Archiv
        nicht_mehr_online = archiv[~archiv["id"].isin(heute_df["id"])]
        neu = pd.concat([nicht_mehr_online, heute_df], ignore_index=True)

    # 12-Monate-Grenze: was seit über einem Jahr nicht mehr online war, fliegt raus
    grenze = (date.fromisoformat(heute) - timedelta(days=AUFBEWAHREN_TAGE)).isoformat()
    neu = neu[neu["zuletzt_gesehen"] >= grenze]

    # Datumsspalten nach vorne, neueste Inserate zuoberst
    spalten = ["id", "zuerst_gesehen", "zuletzt_gesehen"]
    neu = neu[spalten + [s for s in neu.columns if s not in spalten]]
    return neu.sort_values(["zuerst_gesehen", "id"], ascending=False)


def main():
    heute = date.today().isoformat()
    archiv = pd.read_csv(ARCHIV) if os.path.exists(ARCHIV) else pd.DataFrame()
    vorher = len(archiv)

    print("Lade Inserate von Flatfox ...")
    heute_df = hole_heutige_inserate()
    if heute_df.empty:
        # Lieber nichts speichern, als das Archiv mit einer leeren Liste zu überschreiben
        raise SystemExit("Keine Inserate erhalten, Archiv bleibt unverändert.")

    archiv = aktualisiere_archiv(archiv, heute_df, heute)
    archiv.to_csv(ARCHIV, index=False)

    neu = (archiv["zuerst_gesehen"] == heute).sum()
    print(f"Heute online im Kanton SG: {len(heute_df)} Inserate, davon {neu} neu.")
    print(f"Archiv: vorher {vorher}, jetzt {len(archiv)} Inserate.")


if __name__ == "__main__":
    main()