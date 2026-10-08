"""
Holt alle Miet-Inserate von Flatfox im Kanton St. Gallen und speichert sie
als Excel-Tabelle.

Benötigt: Python 3 und die Pakete requests, pandas, openpyxl
    pip install requests pandas openpyxl

Starten:
    python flatfox_sg.py

Ergebnis: Datei "flatfox_sg_<datum>.xlsx" im selben Ordner.
"""

import time
from datetime import date

import pandas as pd
import requests

FLATFOX_URL = "https://flatfox.ch/api/v1/public-listing/"
PLZ_URL = "https://openplzapi.org/ch/Cantons/17/Localities"  # 17 = Kanton St. Gallen

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
    print(f"{len(plz)} Postleitzahlen im Kanton St. Gallen geladen.")
    return plz


def lade_flatfox_inserate():
    """Holt alle Inserate der Schweiz, Seite für Seite (die API kann nicht nach Kanton filtern)."""
    inserate = []
    url = FLATFOX_URL
    params = {"limit": 500}
    while url:
        antwort = requests.get(url, params=params, timeout=60)
        antwort.raise_for_status()
        daten = antwort.json()
        inserate.extend(daten["results"])
        print(f"  {len(inserate)} von {daten['count']} Inseraten geladen ...")
        url = daten["next"]  # Link zur nächsten Seite, None auf der letzten Seite
        params = None        # steckt ab jetzt schon im "next"-Link
        time.sleep(0.5)      # Server nicht überlasten
    return inserate


def merkmal_namen(attributes):
    """Ausstattung kann als Liste von Texten oder von {"name": ...} kommen."""
    namen = set()
    for a in attributes or []:
        namen.add(a["name"] if isinstance(a, dict) else a)
    return namen


def main():
    sg_plz = lade_sg_postleitzahlen()
    print("Lade Inserate von Flatfox (das dauert ein paar Minuten) ...")
    alle = lade_flatfox_inserate()

    zeilen = []
    for i in alle:
        # Nur Mietwohnungen und Häuser im Kanton SG
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
    if df.empty:
        print("Keine Inserate im Kanton St. Gallen gefunden.")
        return

    # Miete pro m² (netto) als zusätzliche Spalte, nur wo beides vorhanden ist
    df["miete_pro_m2"] = (df["miete_netto"] / df["flaeche_m2"]).round(2)

    datei = f"flatfox_sg_{date.today().isoformat()}.xlsx"
    df.to_excel(datei, index=False)

    print()
    print(f"{len(df)} Inserate im Kanton St. Gallen gespeichert in {datei}")
    print(f"  davon mit Nettomiete:      {df['miete_netto'].notna().sum()}")
    print(f"  davon mit Fläche:          {df['flaeche_m2'].notna().sum()}")
    print(f"  davon mit Baujahr:         {df['baujahr'].notna().sum()}")
    print(f"  Median Nettomiete pro m²:  {df['miete_pro_m2'].median()} CHF")


if __name__ == "__main__":
    main()