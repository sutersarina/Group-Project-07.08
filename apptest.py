"""
Streamlit-Version: Holt alle Miet-Inserate von Flatfox im Kanton St. Gallen,
zeigt sie als Tabelle an und bietet sie als Excel-Download an.

Lokal starten:      streamlit run streamlit_app.py
Streamlit Cloud:    diese Datei + requirements.txt ins GitHub-Repo legen
"""

import io
import time
from datetime import date

import pandas as pd
import requests
import streamlit as st

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
    return plz


def lade_flatfox_inserate(fortschritt):
    """Holt alle Inserate der Schweiz, Seite für Seite (die API kann nicht nach Kanton filtern)."""
    inserate = []
    url = FLATFOX_URL
    params = {"limit": 500}
    while url:
        antwort = requests.get(url, params=params, timeout=60)
        antwort.raise_for_status()
        daten = antwort.json()
        inserate.extend(daten["results"])
        fortschritt.progress(
            min(len(inserate) / max(daten["count"], 1), 1.0),
            text=f"{len(inserate)} von {daten['count']} Inseraten geladen ...",
        )
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


def baue_tabelle(alle, sg_plz):
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
    if not df.empty:
        # Miete pro m² (netto), nur wo beides vorhanden ist
        df["miete_pro_m2"] = (df["miete_netto"] / df["flaeche_m2"]).round(2)
    return df


st.title("Flatfox-Inserate Kanton St. Gallen")
st.write("Lädt alle aktuellen Mietinserate von Flatfox im Kanton St. Gallen.")

if st.button("Daten laden"):
    with st.spinner("Lade Postleitzahlen ..."):
        sg_plz = lade_sg_postleitzahlen()
    fortschritt = st.progress(0.0, text="Lade Inserate ...")
    alle = lade_flatfox_inserate(fortschritt)
    st.session_state["df"] = baue_tabelle(alle, sg_plz)

if "df" in st.session_state:
    df = st.session_state["df"]
    if df.empty:
        st.warning("Keine Inserate im Kanton St. Gallen gefunden.")
    else:
        spalte1, spalte2, spalte3 = st.columns(3)
        spalte1.metric("Inserate", len(df))
        spalte2.metric("mit Nettomiete und Fläche", int(df["miete_pro_m2"].notna().sum()))
        spalte3.metric("Median CHF/m²", df["miete_pro_m2"].median())

        st.dataframe(df)

        puffer = io.BytesIO()
        df.to_excel(puffer, index=False)
        st.download_button(
            "Als Excel herunterladen",
            data=puffer.getvalue(),
            file_name=f"flatfox_sg_{date.today().isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )