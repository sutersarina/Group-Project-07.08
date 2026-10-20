# RentCheck St. Gallen

Group project 07.08 – Fundamentals and Methods of Computer Science, University of St. Gallen (HSG).
Team: Riana, Sarina, Noreen.

RentCheck tells you whether a rent in the canton of St. Gallen is cheap, fair or expensive.
It compares the rent with real rental listings from [Flatfox](https://flatfox.ch) and uses
machine learning to estimate what an apartment should cost.

## Features

- **RentCheck (for tenants):** enter town, rooms, living area, year built and your gross rent.
  The app estimates the expected rent and rates yours as a bargain, fair or rather expensive.
- **Map:** the 8 regions (electoral districts) of the canton, colored by average price per m².
  Optionally shows every single apartment. Clicking a region selects it.
- **Market comparison:** shows what share of similar apartments are cheaper than yours,
  and lists similar apartments that are online on Flatfox right now.
- **For landlords:** estimates a possible rent with a range, and shows how the year built
  affects the rent.
- **Join in:** users can save their own rent anonymously (one entry per visit). The averages
  and the model are recalculated with every new entry.

## How it works

1. **Data collection** – `sammeln.py` downloads all current Flatfox listings in the canton
   of St. Gallen and adds them to `archiv.csv` (listings from the last 12 months).
   A GitHub Action (`.github/workflows/archiv_taeglich.yml`) runs it every day.
2. **Cleaning** – the app keeps normal apartments only (no houses, no furnished or temporary
   apartments) and removes implausible listings (below CHF 8 or above CHF 60 per m²).
   All rents are gross rents including utilities.
3. **Machine learning** – two scikit-learn models are trained on region, rooms, area and
   year built: a linear regression and a random forest. Both are evaluated with 5-fold
   cross-validation, and the app uses the one with the smaller average error.
4. **User entries** – saved rents are stored in a local SQLite database (`rentcheck.db`)
   and added to the training data.

## Run it locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_sg.py
```

To update the listings archive by hand: `python sammeln.py`.

## Files

| File | Purpose |
| --- | --- |
| `app_sg.py` | The RentCheck app |
| `sammeln.py` | Downloads the Flatfox listings and updates `archiv.csv` |
| `archiv.csv` | Archive of the listings (updated daily) |
| `kanton_sg.geojson` | Border of the canton (used to filter listings and draw the map) |
| `streamlit_app.py` | Simple viewer and download of the archive |

## Limitations

- About two thirds of the listings do not state the year built; for those, the median
  year of the other listings is used.
- Listings show asking rents, which can differ from rents in existing contracts.
- On Streamlit Community Cloud, the SQLite file is reset whenever the app restarts, so saved
  user entries are not kept permanently there.
- The result is a rough guideline and not legal advice.
