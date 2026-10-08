"""
Streamlit-Version: Zeigt das Archiv der Flatfox-Mietinserate im Kanton St. Gallen
(alle Inserate der letzten 12 Monate) als Tabelle an und bietet es als Download an.

Das Archiv (archiv.csv) wird jeden Tag von GitHub Actions mit sammeln.py
aktualisiert. Gibt es noch kein Archiv, lädt die App die heutigen Inserate
direkt von Flatfox.

Lokal starten:      streamlit run streamlit_app.py
Streamlit Cloud:    diese Datei, sammeln.py und requirements.txt ins GitHub-Repo legen
"""

import io
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

from sammeln import ARCHIV, hole_heutige_inserate


@st.cache_data(show_spinner=False)
def lade_archiv(geaendert_um):
    """Liest archiv.csv. geaendert_um sorgt dafür, dass die Datei neu gelesen
    wird, sobald GitHub Actions eine neue Version gespeichert hat."""
    return pd.read_csv(ARCHIV)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def lade_live():
    """Notlösung, solange es noch kein Archiv gibt: heutige Inserate direkt von Flatfox."""
    df = hole_heutige_inserate()
    heute = date.today().isoformat()
    df.insert(1, "zuerst_gesehen", heute)
    df.insert(2, "zuletzt_gesehen", heute)
    return df


def lade_daten():
    if os.path.exists(ARCHIV):
        geaendert_um = os.path.getmtime(ARCHIV)
        stand = datetime.fromtimestamp(geaendert_um).strftime("%d.%m.%Y %H:%M")
        return lade_archiv(geaendert_um), f"Archiv vom {stand}"
    return lade_live(), "live von Flatfox (noch kein Archiv vorhanden)"


st.title("Flatfox-Inserate Kanton St. Gallen")
st.write("Alle Mietinserate von Flatfox im Kanton St. Gallen der letzten 12 Monate.")

with st.spinner("Lade Inserate ..."):
    df, stand = lade_daten()

st.caption(f"Datenstand: {stand}")
rolle = st.radio("Ich bin ...", ["Mieter/in", "Vermieter/in"], horizontal=True)

with st.sidebar:
    st.header("Deine Wohnung")
    gemeinde = st.selectbox("Gemeinde", sorted(df["gemeinde"].dropna().unique()))
    flaeche = st.number_input("Wohnfläche (m²)", min_value=15, max_value=300, value=70)
    zimmer = st.select_slider("Zimmer", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], value=3.5)
    baujahr = st.number_input("Baujahr", min_value=1850, max_value=2026, value=1990)
    if rolle == "Mieter/in":
        meine_miete = st.number_input("Deine Nettomiete (CHF)", min_value=0, max_value=10000, value=1500)
    pruefen = st.button("Miete prüfen")

if df.empty:
    st.warning("Keine Inserate im Kanton St. Gallen gefunden.")
else:
    heute_online = df["zuletzt_gesehen"] == df["zuletzt_gesehen"].max()

    spalte1, spalte2, spalte3 = st.columns(3)
    spalte1.metric("Inserate im Archiv", len(df))
    spalte2.metric("davon aktuell online", int(heute_online.sum()))
    spalte3.metric("Median CHF/m²", df["miete_pro_m2"].median())

    if st.checkbox("Nur aktuell online"):
        df = df[heute_online]

    st.dataframe(df)

    # CSV geht immer und lässt sich auch in Excel öffnen
    st.download_button(
        "Als CSV herunterladen (öffnet in Excel)",
        data=df.to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name=f"flatfox_sg_{date.today().isoformat()}.csv",
        mime="text/csv",
    )
    try:
        puffer = io.BytesIO()
        df.to_excel(puffer, index=False)
        st.download_button(
            "Als Excel herunterladen",
            data=puffer.getvalue(),
            file_name=f"flatfox_sg_{date.today().isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.info("Für den Excel-Download fehlt das Paket openpyxl: im Terminal `pip install openpyxl` ausführen.")
