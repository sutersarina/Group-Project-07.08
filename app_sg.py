# =====================================================================
# RentCheck St. Gallen – Streamlit app
# Compares your own rent with average rents of the regions in the
# canton of St. Gallen, based on real Flatfox listings (archiv.csv).
# Three views: Home, RentCheck (for tenants) and Landlords. RentCheck
# compares by region, rooms, area and year built; a machine learning
# model estimates the expected rent.
# Users can save their rent anonymously (SQLite); averages and the
# model are then recalculated automatically. The landlord page uses
# the same model to estimate a possible rent for an apartment.
# =====================================================================

# ---------- Import libraries ----------
import json                     # Reading the canton border (GeoJSON)
import sqlite3                  # Database (built into Python)
from datetime import datetime   # Timestamp for saved entries
from pathlib import Path        # File paths

import streamlit as st          # Web app framework (user interface)
import pandas as pd             # Tables / data (DataFrames)
import altair as alt            # Charts
import plotly.express as px     # Interactive map
import plotly.graph_objects as go   # Extra map layer (single apartments)

# Machine learning (scikit-learn)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------- Page settings ----------
# initial_sidebar_state="expanded": the menu (sidebar on the left) is open at the start
st.set_page_config(page_title="RentCheck St. Gallen", page_icon="🏠", layout="centered",
                   initial_sidebar_state="expanded")

# ---------- Design (CSS) ----------
# This defines the look of the app (font, colors, cards etc.)
st.markdown("""
<style>
/* Load the Inter font from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

/* Apply the font to the whole app */
html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit default elements (menu, footer, toolbar).
   The header stays (transparent) so the menu button at the top left remains visible. */
#MainMenu, footer {visibility: hidden;}
[data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden;}
header {background: transparent !important;}

/* Top spacing and maximum width of the content */
.block-container {padding-top: 2rem; max-width: 900px;}

/* Hero: large title area at the top of the page */
.hero {text-align: center; padding: 3.5rem 1rem 2.5rem;}
/* Smaller variant of the hero (for the sub pages) */
.hero.small {padding: 1.5rem 1rem 1rem;}
/* Small label above the title ("CS PROJECT · HSG") */
.badge {
    display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
    background: rgba(99, 102, 241, 0.12); color: #6366f1;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em;
}
/* Main title with gradient */
.hero h1 {
    font-size: 3.6rem; font-weight: 800; line-height: 1.1; margin: 1rem 0 0.8rem;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero.small h1 {font-size: 2.4rem;}
/* Subtitle below the main title */
.hero p {font-size: 1.15rem; opacity: 0.7; max-width: 560px; margin: 0 auto;}

/* Section titles: small label and large title */
.section-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #a855f7; margin: 3rem 0 0.3rem;
}
.section-title {font-size: 1.9rem; font-weight: 700; margin-bottom: 1.2rem;}
/* Top spacing when a section has no label */
.section-title.no-label {margin-top: 3rem;}

/* Cards (problem, solution, features, team) */
.card {
    border: 1px solid rgba(128, 128, 128, 0.18); border-radius: 18px;
    padding: 1.5rem; height: 100%; background: rgba(128, 128, 128, 0.04);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
/* Card lifts slightly when the mouse hovers over it */
.card:hover {transform: translateY(-4px); border-color: rgba(168, 85, 247, 0.5);}
.card .icon {font-size: 1.8rem;}
.card h4 {margin: 0.6rem 0 0.4rem; font-weight: 600;}
.card p {opacity: 0.7; font-size: 0.95rem; margin: 0;}

/* Steps with a numbered circle */
.step {display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1rem;}
.step-num {
    min-width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #a855f7); color: white;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
}
.step p {margin: 0; opacity: 0.8; padding-top: 0.4rem;}

/* Result box (cheap / fair / expensive) */
.result {
    border-radius: 18px; padding: 1.4rem; text-align: center; margin-top: 1rem;
    color: white; font-weight: 600;
}
.result .big {font-size: 2rem; font-weight: 800; display: block;}

/* Team: round avatar with initial */
.avatar {
    width: 72px; height: 72px; border-radius: 50%; margin: 0 auto 0.6rem;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 1.6rem; font-weight: 700;
}
/* Team card: avatar and name centered in the box, one below the other */
.team-card {display: flex; flex-direction: column; align-items: center; justify-content: center;
            text-align: center;}
/* Name below the avatar (no heading, as Streamlit adds a link icon next to headings) */
.team-name {font-size: 1.1rem; font-weight: 600;}
/* Footer at the very bottom */
.footer {text-align: center; opacity: 0.5; font-size: 0.85rem; margin: 4rem 0 1rem;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA: regions, Flatfox listings (archiv.csv), database (SQLite)
# =====================================================================

# The 8 electoral districts (regions) of the canton of St. Gallen with the coordinates
# of their main town (lat/lon are needed for the map)
REGIONS_GEO = pd.DataFrame({
    "Region":    ["St. Gallen", "Rorschach", "Rheintal", "Werdenberg",
                  "Sarganserland", "See-Gaster", "Toggenburg", "Wil"],
    "Main town": ["St. Gallen", "Rorschach", "Altstätten", "Buchs SG",
                  "Sargans", "Uznach", "Wattwil", "Wil SG"],
    "lat":       [47.4245, 47.4786, 47.3772, 47.1658, 47.0487, 47.2246, 47.2987, 47.4616],
    "lon":       [9.3767, 9.4932, 9.5427, 9.4728, 9.4413, 8.9806, 9.0870, 9.0447],
})

# Paths of the database file and of the listings archive (in the same folder as this script)
DB_PATH = Path(__file__).with_name("rentcheck.db")
ARCHIVE_PATH = Path(__file__).with_name("archiv.csv")   # Real Flatfox listings, updated daily by sammeln.py
CANTON_PATH = Path(__file__).with_name("kanton_sg.geojson")   # Border of the canton (for the map)

ROOMS_OPTIONS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]   # Choices in the input fields

# Plausibility limits for rents (CHF per m²), so that typos
# do not distort the average
MIN_PRICE_M2 = 8
MAX_PRICE_M2 = 60

# SQL for creating the table with the user entries
SQL_ENTRIES = """CREATE TABLE IF NOT EXISTS entries (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     region TEXT, rooms REAL, area REAL, year_built REAL, rent REAL,
                     created TEXT)"""


@st.cache_data(show_spinner=False)   # only read the file again when it has changed
def load_listings(modified):
    """Reads the real Flatfox listings from archiv.csv and keeps the usable apartments.
    modified (time of the last change of the file) makes the cache reload the file
    as soon as sammeln.py has updated the archive."""
    raw = pd.read_csv(ARCHIVE_PATH)
    latest = raw["zuletzt_gesehen"].max()     # Date of the last update -> these listings are online

    # Only normal apartments: no houses, no furnished or temporary apartments
    raw = raw[(raw["kategorie"] == "APARTMENT")
              & ~raw["moebliert"].fillna(False).astype(bool)
              & ~raw["befristet"].fillna(False).astype(bool)]

    # Same columns as the user entries; the rent is the gross rent (incl. utilities)
    listings = pd.DataFrame({
        "region":     raw["wahlkreis"],
        "rooms":      raw["zimmer"],
        "area":       raw["flaeche_m2"],
        "year_built": raw["baujahr"].where(raw["baujahr"].between(1800, 2030)),   # 0 etc. -> missing
        "rent":       raw["miete_brutto"],
        # Extra information for the map and the list of similar apartments
        "town":       raw["plz"].astype(int).astype(str) + " " + raw["ort"],
        "lat":        raw["lat"],
        "lon":        raw["lon"],
        "link":       raw["link"],
        "online":     raw["zuletzt_gesehen"] == latest,
    }).dropna(subset=["region", "rooms", "area", "rent"])

    # Remove implausible listings (typos, parking spaces listed as apartments etc.)
    price_m2 = listings["rent"] / listings["area"]
    listings = listings[price_m2.between(MIN_PRICE_M2, MAX_PRICE_M2)
                        & listings["rooms"].between(min(ROOMS_OPTIONS), max(ROOMS_OPTIONS))]
    return listings.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_towns(modified):
    """All postcodes/towns in the archive with their region, e.g. {"9000 St. Gallen": "St. Gallen"}."""
    raw = pd.read_csv(ARCHIVE_PATH, usecols=["plz", "ort", "wahlkreis"]).dropna().drop_duplicates()
    towns = raw["plz"].astype(int).astype(str) + " " + raw["ort"]
    return dict(sorted(zip(towns, raw["wahlkreis"])))


@st.cache_data(show_spinner=False)
def load_canton_outline():
    """Border of the canton of St. Gallen (GeoJSON), drawn as a line on the map."""
    return json.loads(CANTON_PATH.read_text(encoding="utf-8"))


def table_columns(con, table):
    """Returns the column names of a table."""
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def db_init():
    """Creates the database and the table for the user entries (if they do not exist yet)."""
    con = sqlite3.connect(DB_PATH)
    con.execute(SQL_ENTRIES)        # Table with the user entries (anonymous, no personal data)
    con.commit()
    con.close()


def load_all_data():
    """Reads the Flatfox listings and the user entries and returns them as one combined table."""
    listings = load_listings(ARCHIVE_PATH.stat().st_mtime).assign(source="Flatfox")
    con = sqlite3.connect(DB_PATH)
    users = pd.read_sql_query(
        "SELECT region, rooms, area, year_built, rent, 'User' AS source FROM entries", con)
    con.close()
    data = pd.concat([listings, users], ignore_index=True)
    # Many listings do not state the year built -> use the median of the known years
    data["year_built"] = data["year_built"].fillna(data["year_built"].median())
    return data


def compute_regions(data):
    """Computes the average price per m² for each region (from listings + user entries)."""
    data = data.assign(price_m2=data["rent"] / data["area"])    # Price per m² of each apartment
    stats = (data.groupby("region")
             .agg(price=("price_m2", "mean"),                               # Average
                  count=("price_m2", "size"),                               # Number of entries
                  users=("source", lambda s: int((s == "User").sum())))     # of which from users
             .reset_index().rename(columns={"region": "Region"}))
    # Merge with the coordinates (the order of the regions stays the same)
    regions = REGIONS_GEO.merge(stats, on="Region", how="left")
    regions["CHF per m²"] = regions["price"].round(1)
    regions["Entries"] = regions["count"]
    regions["of which users"] = regions["users"]
    return regions.drop(columns=["price", "count", "users"])


def region_average(region):
    """Current average price per m² of a region (CHF)."""
    data = load_all_data()
    d = data[data["region"] == region]
    return float((d["rent"] / d["area"]).mean())


def save_entry():
    """Saves the user's inputs in the database (called by the button)."""
    # Read the values from the input fields
    region = st.session_state["region"]
    rooms = float(st.session_state["rooms"])
    area = float(st.session_state["area"])
    year_built = float(st.session_state["year_built"])
    rent = float(st.session_state["rent"])

    # Plausibility check: do not save unrealistic values
    price_m2 = rent / area
    if not (MIN_PRICE_M2 <= price_m2 <= MAX_PRICE_M2):
        st.session_state["message"] = (
            "warning",
            f"That is CHF {price_m2:.0f} per m² – this looks unrealistic and was not "
            f"saved. Please check the area and the rent.")
        return

    before = region_average(region)     # Average before saving

    # Write the entry to the database
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO entries (region, rooms, area, year_built, rent, created) "
                "VALUES (?,?,?,?,?,?)",
                (region, rooms, area, year_built, rent,
                 datetime.now().isoformat(timespec="seconds")))
    con.commit()
    con.close()
    st.session_state["saved"] = True    # Only one entry per visit (protection against spam)

    after = region_average(region)      # Average after saving
    st.session_state["message"] = (
        "ok",
        f"Thank you! Your rent was saved anonymously. New average in {region}: "
        f"CHF {after:.2f} per m² (before: {before:.2f}).")


# =====================================================================
# MACHINE LEARNING: estimate the rent
# =====================================================================
@st.cache_data(show_spinner="Training the models …")   # only retrain when the data changes
def train_model(data):
    """Trains two models (linear regression and random forest), measures their accuracy
    and returns the more accurate one together with a comparison table."""
    X = data[["region", "rooms", "area", "year_built"]]   # Inputs (features)
    y = data["rent"]                                      # Target value: the rent

    # The two candidates: a straight-line model and a model made of many decision trees
    candidates = {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
    }

    # Measure accuracy with 5-fold cross-validation:
    # the model always predicts apartments it has not seen during training
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, regressor in candidates.items():
        # Region is text -> convert to numbers with one-hot encoding, rooms/area/year stay as they are
        preprocessing = ColumnTransformer(
            [("region", OneHotEncoder(handle_unknown="ignore"), ["region"])],
            remainder="passthrough")
        model = Pipeline([("preprocessing", preprocessing), ("regression", regressor)])
        predictions = cross_val_predict(model, X, y, cv=cv)
        mae = mean_absolute_error(y, predictions)     # average error in CHF
        r2 = r2_score(y, predictions)                 # coefficient of determination (1 = perfect)
        results[name] = (model, mae, r2, predictions)

    # Keep the model with the smaller average error
    best = min(results, key=lambda name: results[name][1])
    model, mae, r2, predictions = results[best]
    model.fit(X, y)                               # Train the final model on all data

    scores = pd.DataFrame({"Model": list(results),
                           "Avg. error (CHF)": [round(r[1]) for r in results.values()],
                           "R²": [round(r[2], 2) for r in results.values()]})
    return model, mae, r2, predictions, best, scores


# =====================================================================
# HELPER FUNCTIONS FOR THE USER INTERFACE
# =====================================================================
def card(icon, title, text):
    """Builds the HTML code for a card with icon, title and text."""
    return f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'


def section(label, title):
    """Shows a section title. If label is empty, only the title is shown."""
    if label:
        # With a small label above the title
        st.markdown(f'<div class="section-label">{label}</div>'
                    f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    else:
        # Title only, without a label
        st.markdown(f'<div class="section-title no-label">{title}</div>',
                    unsafe_allow_html=True)


def clear_town(town_key):
    """Empties the town search when the region is changed by hand (the town would no longer match)."""
    st.session_state.pop(town_key, None)


def town_picker(region_key):
    """Search field for postcode/town; choosing a town sets the matching region."""
    towns = load_towns(ARCHIVE_PATH.stat().st_mtime)
    town_key = f"{region_key}_town"

    def adopt_region():
        # Called when a town is chosen: set the region select box to the town's region
        town = st.session_state[town_key]
        if town:
            st.session_state[region_key] = towns[town]

    st.selectbox("Town or postcode (optional)", list(towns), index=None, key=town_key,
                 placeholder="Type e.g. 9000 or Wil …", on_change=adopt_region,
                 help="Not sure which region you live in? Choose your town and the "
                      "region is set automatically.")


def similar_listings(data, region, rooms):
    """Flatfox apartments similar to the input: same region and at most ½ room more or less.
    If there are too few in the region, the whole canton is used.
    Returns the apartments and a short description of the selection."""
    flatfox = data[data["source"] == "Flatfox"]
    near = flatfox[(flatfox["rooms"] - rooms).abs() <= 0.5]
    rooms_text = f"{max(1, rooms - 0.5):g}–{rooms + 0.5:g} rooms"
    in_region = near[near["region"] == region]
    if len(in_region) >= 10:
        return in_region, f"{rooms_text} in {region}"
    return near, f"{rooms_text} in the whole canton"


def listing_table(similar, area):
    """Shows the similar apartments that are online right now, the closest in size first."""
    online = similar[similar["online"].eq(True)]
    online = online.assign(size_diff=(online["area"] - area).abs()).sort_values("size_diff").head(8)
    if online.empty:
        st.caption("No similar apartments are online right now.")
        return
    table = pd.DataFrame({"Town": online["town"], "Rooms": online["rooms"],
                          "Area (m²)": online["area"], "Gross rent (CHF)": online["rent"],
                          "CHF per m²": (online["rent"] / online["area"]).round(1),
                          "Listing": online["link"]})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Rooms": st.column_config.NumberColumn(format="%g"),
        "Area (m²)": st.column_config.NumberColumn(format="%.0f"),
        "Gross rent (CHF)": st.column_config.NumberColumn(format="%.0f"),
        "Listing": st.column_config.LinkColumn(display_text="Open on Flatfox"),
    })


def go_to(page):
    """Switches the view (called by the buttons as a callback)."""
    # Save the chosen page in the session state ("home", "compare" or "landlord")
    st.session_state["page"] = page
    # Reset the map selection so that a new click is recognized again
    st.session_state.pop("last_map_selection", None)


def reset_map_selection():
    """Called when switching in the menu (resets the map selection)."""
    st.session_state.pop("last_map_selection", None)


def navigation():
    """Menu in the sidebar (on the left, can be opened/closed with the icon at the top left)."""
    # Names of the pages as shown in the menu
    names = {"home": "🏠 Home", "compare": "📊 RentCheck",
             "landlord": "🏢 For landlords"}
    with st.sidebar:
        st.markdown("### 🏠 RentCheck")   # Title in the menu
        # Page selection; key="page" links it to the stored page state,
        # so the menu and the buttons on the pages always stay in sync
        st.radio("Choose page", list(names.keys()), format_func=lambda p: names[p],
                 key="page", on_change=reset_map_selection,
                 label_visibility="collapsed")


# =====================================================================
# VIEW 1: HOME
# =====================================================================
def page_home():
    # ----- Hero: title and short explanation -----
    st.markdown("""
    <div class="hero">
        <span class="badge">CS PROJECT · HSG</span>
        <h1>Is your rent fair?</h1>
        <p>RentCheck compares rents in the canton of St. Gallen with real rental
        listings from Flatfox – in a few seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----- Two buttons: for tenants and for landlords -----
    _, b1, b2, _ = st.columns([0.4, 1.3, 1.3, 0.4])   # outer columns provide spacing
    b1.button("Compare my rent now →", key="cta_start", type="primary",
              on_click=go_to, args=("compare",))
    b2.button("I am a landlord →", key="cta_landlord",
              on_click=go_to, args=("landlord",))

    # ----- Problem & solution (two cards next to each other) -----
    section("The idea", "Why RentCheck?")
    c1, c2 = st.columns(2)
    c1.markdown(card("😩", "The problem",
                     "Rents differ a lot by region, size and location. "
                     "It is often hard to know whether an offer is fair."), unsafe_allow_html=True)
    c2.markdown(card("💡", "Our solution",
                     "A web app that compares your offer with real average values "
                     "and clearly tells you: cheap, fair or expensive."), unsafe_allow_html=True)

    # ----- Features (three cards next to each other) -----
    section("", "Features")   # only the title "Features", without a label
    f1, f2, f3 = st.columns(3)
    f1.markdown(card("📊", "Comparison", "Rate your rent by region, rooms, area and year built – "
                     "and contribute your own rent anonymously"), unsafe_allow_html=True)
    f2.markdown(card("🗺️", "Regions", "The 8 regions of the canton of St. Gallen on the map, "
                     "colored by price per m² – and similar apartments online right now"),
                unsafe_allow_html=True)
    f3.markdown(card("🤖", "Prediction", "An ML model estimates what an apartment should cost – "
                     "also for landlords"), unsafe_allow_html=True)

    # ----- How it works: three numbered steps -----
    section("How it works", "How does it work?")
    for i, text in enumerate([
        "You search your town (or click your region on the map) and enter rooms, living area "
        "and rent.",
        "A machine learning model estimates from real Flatfox listings what the apartment "
        "should cost.",
        "You immediately see whether your offer is cheap, fair or expensive – and how it "
        "compares with similar apartments on the market.",
    ], start=1):
        st.markdown(f'<div class="step"><div class="step-num">{i}</div><p>{text}</p></div>',
                    unsafe_allow_html=True)

    # ----- Team: one card per person with initial -----
    section("Team", "Who is behind it")
    team = ["Riana", "Sarina", "Noreen"]
    for col, name in zip(st.columns(3), team):
        col.markdown(f'<div class="card team-card">'
                     f'<div class="avatar">{name[0]}</div><div class="team-name">{name}</div></div>',
                     unsafe_allow_html=True)


# =====================================================================
# VIEW 2: RENTCHECK (map, comparison by rooms, ML estimate, saving)
# =====================================================================
def page_compare():
    # Load the current data and recompute the averages per region
    data = load_all_data()
    regions = compute_regions(data)

    # ----- Page title -----
    st.markdown("""
    <div class="hero small">
        <h1>Compare your rent</h1>
        <p>Choose your region, enter rooms, area, year built and rent, and see how fair your rent is.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----- Interactive map -----
    section("Map", "Regions in the canton of St. Gallen")
    st.caption("Click a point to select the region. "
               "Zoom and pan with the mouse or by touch.")

    # Remember the currently selected region (default: St. Gallen)
    if "region" not in st.session_state:
        st.session_state["region"] = "St. Gallen"

    # Newer Plotly versions call the function scatter_map, older ones scatter_mapbox
    new_plotly = hasattr(px, "scatter_map")
    scatter = px.scatter_map if new_plotly else px.scatter_mapbox
    map_kwargs = {"map_style": "open-street-map"} if new_plotly \
        else {"mapbox_style": "open-street-map"}

    # Option: additionally show every single apartment as a small point
    show_listings = st.toggle("Show individual apartments", key="show_listings")

    # Create the map: one point per region (position via lat/lon),
    # colored by the price per m²; the selected region is drawn larger
    fig = scatter(
        regions,
        lat="lat", lon="lon",
        hover_name="Region",                 # Title when hovering
        hover_data={"Main town": True, "CHF per m²": True, "Entries": True,
                    "lat": False, "lon": False},
        text="Region",                       # Label at the point
        color="CHF per m²",                  # Color = average price per m²
        color_continuous_scale=["#c7c9f5", "#a855f7", "#ec4899"],   # cheap = light, expensive = pink
        range_color=[regions["CHF per m²"].min(), regions["CHF per m²"].max()],
        size=[3 if r == st.session_state["region"] else 1 for r in regions["Region"]],
        size_max=26,
        zoom=8.2,                            # Zoom level at the start
        center={"lat": 47.25, "lon": 9.25},  # Map center (canton of St. Gallen)
        height=480,
        **map_kwargs,
    )
    fig.update_traces(textposition="top center")

    # Layer with the single apartments (same color scale as the regions)
    if show_listings:
        flats = data[data["source"] == "Flatfox"]
        flats = flats.assign(price_m2=(flats["rent"] / flats["area"]).round(1))
        trace = (go.Scattermap if new_plotly else go.Scattermapbox)(
            lat=flats["lat"], lon=flats["lon"], mode="markers",
            marker=dict(size=7, opacity=0.6, color=flats["price_m2"], coloraxis="coloraxis"),
            text=flats["town"], customdata=flats[["rooms", "area", "rent", "price_m2"]],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:g} rooms, %{customdata[1]:.0f} m²"
                          "<br>CHF %{customdata[2]:,.0f} (CHF %{customdata[3]} per m²)<extra></extra>")
        fig.add_trace(trace)
        fig.data = fig.data[::-1]            # draw the apartments below the region points
    region_curve = len(fig.data) - 1         # The region points are the last layer

    # Border of the canton as a line, no margins, legend hidden, small color bar
    fig.update_layout({"map" if new_plotly else "mapbox": {"layers": [
                          {"source": load_canton_outline(), "type": "line",
                           "color": "#6366f1", "line": {"width": 2}}]}},
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      coloraxis_colorbar=dict(title="CHF/m²", thickness=12))

    # Show the map; clicking a point reloads the app
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                            key="sg_map", width="stretch")

    # Click on a region point -> adopt the region (only if the selection has changed);
    # clicks on single apartments are ignored
    points = [p for p in (event.selection.points if event and event.selection else [])
              if p["curve_number"] == region_curve]
    if points:
        clicked = regions.iloc[points[0]["point_index"]]["Region"]
        if st.session_state.get("last_map_selection") != clicked:
            st.session_state["last_map_selection"] = clicked
            if st.session_state["region"] != clicked:
                st.session_state["region"] = clicked
                clear_town("region_town")   # The searched town no longer matches
                st.rerun()   # Reload the app so the select box shows the new region

    # ----- Comparison: inputs -----
    section("Comparison", "Is your rent fair?")

    # Two columns: inputs and result on the left, chart on the right
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        town_picker("region")                                                                # Search town -> region
        region = st.selectbox("Region", regions["Region"], key="region",
                              on_change=clear_town, args=("region_town",))                   # Choose region
        rooms = st.selectbox("Number of rooms", ROOMS_OPTIONS, index=4, key="rooms")         # Choose rooms
        area = st.slider("Living area (m²)", 20, 250, 70, key="area")                        # Choose area
        year_built = st.slider("Year the building was built", 1900, 2026, 1990, key="year_built")  # Choose year built
        rent = st.number_input("Your gross rent (CHF, incl. utilities)", 300, 10000, 1600, step=50,
                               key="rent")      # Your own rent

    # ----- Expected rent: the ML model estimates it from region, rooms, area and year built -----
    model, mae, r2, predictions, model_name, scores = train_model(data)
    input_row = pd.DataFrame({"region": [region], "rooms": [float(rooms)],
                              "area": [float(area)], "year_built": [float(year_built)]})
    expected = float(model.predict(input_row)[0])   # Estimated rent
    difference = rent - expected                    # Difference to your own rent
    tolerance = 0.10 * expected                     # Up to 10 % deviation counts as fair

    # ----- Rating: expensive, fair or bargain -----
    if difference > tolerance:
        color, label = "linear-gradient(135deg, #f43f5e, #ec4899)", "Rather expensive 😬"
    elif difference < -tolerance:
        color, label = "linear-gradient(135deg, #10b981, #14b8a6)", "Bargain 🎉"
    else:
        color, label = "linear-gradient(135deg, #6366f1, #a855f7)", "Fair price 👍"

    # ----- Show the result box -----
    with left:
        st.markdown(f"""
        <div class="result" style="background:{color}">
            {label}
            <span class="big">{difference:+,.0f} CHF</span>
            expected: CHF {expected:,.0f}
        </div>""", unsafe_allow_html=True)

        # Additionally: average of all apartments with the same number of rooms in this region
        same = data[(data["region"] == region) & (data["rooms"] == float(rooms))]
        if len(same) > 0:
            st.caption(f"Avg. rent for {rooms:g} rooms in {region}: "
                       f"CHF {same['rent'].mean():,.0f} ({len(same)} entries)")

    # ----- Chart: average rent by number of rooms in the selected region -----
    d_region = data[data["region"] == region]
    by_rooms = d_region.groupby("rooms")["rent"].mean().reset_index()    # Avg. rent per number of rooms
    by_rooms["Rooms"] = by_rooms["rooms"].map(lambda z: f"{z:g}")        # Number of rooms as text
    by_rooms["Avg. rent (CHF)"] = by_rooms["rent"].round(0)
    selected = f"{rooms:g}"                                              # Selected number of rooms
    # Order of the room counts on the x axis (including the selected one, if missing)
    order = [f"{z:g}" for z in sorted(set(by_rooms["rooms"]) | {float(rooms)})]
    # Bars: average rent per number of rooms, selected number of rooms highlighted
    bars = (alt.Chart(by_rooms)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(x=alt.X("Rooms:O", sort=order, title="Rooms",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Avg. rent (CHF):Q", axis=alt.Axis(grid=False)),
                    color=alt.condition(alt.datum.Rooms == selected,
                                        alt.value("#a855f7"), alt.value("#c7c9f5")),
                    tooltip=["Rooms", "Avg. rent (CHF)"]))
    # Pink marker: your own rent
    own = pd.DataFrame({"Rooms": [selected], "Your rent (CHF)": [rent]})
    marker = (alt.Chart(own)
              .mark_point(shape="diamond", size=220, filled=True, color="#ec4899")
              .encode(x=alt.X("Rooms:O", sort=order),
                      y="Your rent (CHF):Q",
                      tooltip=["Your rent (CHF)"]))
    with right:
        st.altair_chart((bars + marker).properties(height=340), width="stretch")
        st.caption(f"Avg. rent by number of rooms in {region}. The pink marker is your rent.")

    # Info about the data basis (number of entries in total and from users)
    user_count = int((data["source"] == "User").sum())
    st.caption(f"Data basis: {len(data)} apartments – rental listings from Flatfox "
               f"(updated daily) and {user_count} entries from users. All rents are gross rents "
               f"incl. utilities.")

    # ----- Market: where does your rent stand among similar apartments? -----
    section("Market", "How do you compare with similar apartments?")
    similar, scope = similar_listings(data, region, float(rooms))
    if similar.empty:
        st.caption("There are no similar apartments in the data yet.")
    else:
        share = (similar["rent"] < rent).mean() * 100     # Share of similar apartments that are cheaper
        st.write(f"Your rent is higher than **{share:.0f}%** of {len(similar)} similar "
                 f"apartments ({scope}).")
        # Histogram: how many similar apartments cost how much; pink line = your rent
        histogram = (alt.Chart(similar)
                     .mark_bar(color="#c7c9f5", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                     .encode(x=alt.X("rent:Q", bin=alt.Bin(maxbins=30), title="Gross rent (CHF)"),
                             y=alt.Y("count()", title="Apartments", axis=alt.Axis(grid=False)),
                             tooltip=[alt.Tooltip("count()", title="Apartments")]))
        own_line = (alt.Chart(pd.DataFrame({"rent": [rent]}))
                    .mark_rule(color="#ec4899", strokeWidth=3)
                    .encode(x="rent:Q"))
        st.altair_chart((histogram + own_line).properties(height=260), width="stretch")
        st.caption("Gross rents of similar apartments. The pink line is your rent.")

        # Similar apartments that are on Flatfox right now
        st.markdown("**Similar apartments online right now**")
        listing_table(similar, area)

    # ----- Save your own rent -----
    section("Join in", "Save your rent anonymously")
    st.write("With your entry, the averages and the estimate become more accurate. "
             "Only region, rooms, area, year built and rent from the fields above are saved – "
             "no personal data.")
    # Button: calls save_entry() on click; only one entry per visit
    already_saved = st.session_state.get("saved", False)
    st.button("💾 Save my rent", key="save", type="primary",
              on_click=save_entry, disabled=already_saved)
    # Show feedback after saving (only once)
    message = st.session_state.pop("message", None)
    if message:
        kind, text = message
        (st.success if kind == "ok" else st.warning)(text)
    elif already_saved:
        st.caption("You have already saved your rent during this visit – thank you!")

    # ----- Comparison of all regions: price per m² -----
    section("Regions", "Price per m² compared")
    chart = (
        alt.Chart(regions)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)   # Bars with rounded corners
        .encode(
            x=alt.X("Region", sort="-y", axis=alt.Axis(labelAngle=-35, title=None)),  # Sorted by price
            y=alt.Y("CHF per m²", axis=alt.Axis(grid=False)),
            # Selected region dark, others light
            color=alt.condition(alt.datum.Region == region,
                                alt.value("#a855f7"), alt.value("#c7c9f5")),
            # Info when hovering
            tooltip=["Region", "Main town", "CHF per m²", "Entries", "of which users"],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")

    # ----- Accuracy of the ML model -----
    section("Machine learning", "How accurate is the estimate?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg. error (MAE)", f"CHF {mae:,.0f}",
              help="Average difference between estimated and actual rent.")
    m2.metric("Coefficient of determination (R²)", f"{r2:.2f}",
              help="1.00 = perfect prediction, 0 = no better than the average.")
    m3.metric("Training data", f"{len(data)}", f"{user_count} from users",
              delta_color="off")
    st.caption(f"Model used: {model_name} (the more accurate of two models, see details).")

    # Details about the model (chart and explanation), expandable
    with st.expander("Model details"):
        # Chart: estimated vs. actual rent (the closer to the diagonal, the better)
        comparison = pd.DataFrame({"Actual": data["rent"], "Estimated": predictions,
                                   "Region": data["region"], "Source": data["source"]})
        bounds = [float(comparison[["Actual", "Estimated"]].min().min()),
                  float(comparison[["Actual", "Estimated"]].max().max())]
        points_chart = (alt.Chart(comparison)
                        .mark_circle(size=60, opacity=0.6)
                        .encode(x=alt.X("Actual", scale=alt.Scale(domain=bounds),
                                        title="Actual rent (CHF)"),
                                y=alt.Y("Estimated", scale=alt.Scale(domain=bounds),
                                        title="Estimated rent (CHF)"),
                                color=alt.Color("Source", scale=alt.Scale(
                                    domain=["Flatfox", "User"], range=["#a855f7", "#ec4899"])),
                                tooltip=["Region", "Actual", "Estimated", "Source"]))
        diagonal = (alt.Chart(pd.DataFrame({"x": bounds, "y": bounds}))
                    .mark_line(color="gray", strokeDash=[4, 4]).encode(x="x", y="y"))
        st.altair_chart((diagonal + points_chart).properties(height=360), width="stretch")
        st.caption("Each point is an apartment. The estimate comes from cross-validation: "
                   "the model predicts apartments it has not seen during training.")
        st.write(
            "Two models (scikit-learn) learn from the data how region, number of rooms, area "
            "and year built influence the rent: a **linear regression** (one straight-line "
            "formula) and a **random forest** (200 decision trees whose estimates are averaged). "
            "The accuracy is measured with **5-fold cross-validation**: "
            "the data is split into 5 parts, the model learns from 4 parts and is tested on the "
            "remaining part. The app uses the model with the smaller error – currently the "
            f"**{model_name.lower()}**. As soon as someone saves their rent, both models are "
            "automatically retrained with the new data.")
        st.dataframe(scores, hide_index=True, width="stretch")
        st.caption("Many listings do not state the year built; for those, the median year "
                   "of the other apartments is used.")


# =====================================================================
# VIEW 3: FOR LANDLORDS (estimate a possible rent)
# =====================================================================
def page_landlord():
    # Load the current data and train the model (automatically again when there is new data)
    data = load_all_data()
    model, mae, r2, _, model_name, _ = train_model(data)

    # ----- Page title -----
    st.markdown("""
    <div class="hero small">
        <h1>For landlords</h1>
        <p>Enter the key facts of your apartment and see which rent is roughly
        common in the canton of St. Gallen.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----- Inputs about the apartment -----
    section("Your apartment", "Enter the key facts")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        town_picker("l_region")                                                                 # Search town -> region
        region = st.selectbox("Region", REGIONS_GEO["Region"], key="l_region",
                              on_change=clear_town, args=("l_region_town",))                    # Choose region
        rooms = st.selectbox("Number of rooms", ROOMS_OPTIONS, index=4, key="l_rooms")          # Choose rooms
        area = st.slider("Living area (m²)", 20, 250, 70, key="l_area")                         # Choose area
        year_built = st.slider("Year the building was built", 1900, 2026, 1990, key="l_year_built")  # Choose year built

    # ----- Estimate by the ML model -----
    input_row = pd.DataFrame({"region": [region], "rooms": [float(rooms)],
                              "area": [float(area)], "year_built": [float(year_built)]})
    estimate = float(model.predict(input_row)[0])       # Estimated market rent
    # Range: typical model error downwards and upwards, rounded to 10 CHF
    low = round((estimate - mae) / 10) * 10
    high = round((estimate + mae) / 10) * 10

    # ----- Result box -----
    with right:
        st.markdown(f"""
        <div class="result" style="background:linear-gradient(135deg, #6366f1, #a855f7)">
            Possible monthly gross rent
            <span class="big">CHF {estimate:,.0f}</span>
            Range: CHF {low:,.0f} – {high:,.0f}
        </div>""", unsafe_allow_html=True)
        st.caption(f"That is about CHF {estimate / area:.1f} per m². "
                   f"The range corresponds to the typical deviation of the model (± CHF {mae:,.0f}).")

        # For comparison: average of the apartments with the same region and number of rooms
        same = data[(data["region"] == region) & (data["rooms"] == float(rooms))]
        if len(same) > 0:
            st.caption(f"For comparison: avg. rent for {rooms:g} rooms in {region}: "
                       f"CHF {same['rent'].mean():,.0f} ({len(same)} entries).")

    # ----- Chart: how does the possible rent depend on the year built? -----
    section("Influence of the year built", "What does a newer building bring?")
    years = list(range(1900, 2027, 2))                  # Years built for the line
    curve = pd.DataFrame({"region": region, "rooms": float(rooms),
                          "area": float(area), "year_built": [float(j) for j in years]})
    curve["Estimated rent (CHF)"] = model.predict(curve)    # Estimate for every year built
    curve["Year built"] = years
    # Line: estimated rent per year built (all other inputs stay the same)
    line = (alt.Chart(curve)
            .mark_line(color="#a855f7", strokeWidth=3)
            .encode(x=alt.X("Year built:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                    y=alt.Y("Estimated rent (CHF):Q", scale=alt.Scale(zero=False),
                            axis=alt.Axis(grid=False)),
                    tooltip=["Year built", "Estimated rent (CHF)"]))
    # Pink marker: the selected year built
    highlighted = pd.DataFrame({"Year built": [year_built], "Estimated rent (CHF)": [estimate]})
    marker = (alt.Chart(highlighted)
              .mark_point(shape="diamond", size=220, filled=True, color="#ec4899")
              .encode(x="Year built:Q", y="Estimated rent (CHF):Q",
                      tooltip=["Year built", "Estimated rent (CHF)"]))
    st.altair_chart((line + marker).properties(height=320), width="stretch")
    st.caption(f"Estimated rent for {rooms:g} rooms, {area} m² in {region} depending on the year built. "
               f"The pink marker is your building.")

    # ----- Comparable apartments on the market right now -----
    section("Market", "Comparable apartments online right now")
    similar, scope = similar_listings(data, region, float(rooms))
    st.caption(f"Flatfox listings with {scope}, the closest in size first.")
    listing_table(similar, area)

    # ----- Notes -----
    user_count = int((data["source"] == "User").sum())
    st.caption(f"Data basis: {len(data)} apartments – rental listings from Flatfox "
               f"(updated daily) and {user_count} entries from tenants. "
               f"Model: {model_name}, R² = {r2:.2f}.")
    st.info("The result is a rough guideline and not legal or investment advice. "
            "Condition, location, fit-out standard and legal rules on permissible rents are "
            "not taken into account – a professional can help with questions.")


# =====================================================================
# MAIN PROGRAM: show the selected view
# =====================================================================
# Create the database for the user entries
db_init()

# On the first start (or with an invalid value) the home page is active
if st.session_state.get("page") not in ("home", "compare", "landlord"):
    st.session_state["page"] = "home"

# Show the menu in the sidebar
navigation()

# Draw the matching view depending on the selected page
if st.session_state["page"] == "home":
    page_home()
elif st.session_state["page"] == "compare":
    page_compare()
else:
    page_landlord()

# Footer (on all pages)
st.markdown('<div class="footer">Fundamentals and Methods of Computer Science · '
            'University of St. Gallen</div>', unsafe_allow_html=True)