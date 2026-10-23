"""SITR interactive Harris County historical rainfall zones.

Run: python -m streamlit run sitr_map_dashboard_with_risk.py
Install: python -m pip install streamlit pandas pydeck requests shapely

Three illustrative rainfall zones partition Harris County by nearest sampled weather point.
Purple indicates a county-wide NOAA flood report, NOT a confirmed flood in each zone.
"""
from pathlib import Path
from datetime import date, timedelta, datetime
import json
import time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import pydeck as pdk
import requests
from shapely.geometry import shape, mapping, Polygon

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE = DATA / "extended_training_data.csv"
WEATHER = DATA / "multi_location_weather"
PREDICTIONS = DATA / "extended_model_results" / "test_predictions.csv"
BOUNDARY_CACHE = DATA / "harris_county_boundary.geojson"
BOUNDARY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

SITES = [
    ("Houston", 29.7604, -95.3698, "original"),
    ("Katy", 29.7858, -95.8245, "katy"),
    ("Baytown", 29.7355, -94.9774, "baytown"),
]

st.set_page_config(page_title="SITR | Harris County", page_icon="🌧️", layout="wide")
st.title("SITR · Harris County")
st.caption("Historical rainfall exploration and live Houston weather | Educational research demo")
historical_tab, live_tab = st.tabs(["🗺️ Historical Flood Map", "☁️ Live Houston Weather"])

@st.cache_data(show_spinner=False)
def read_base():
    if not BASE.exists():
        raise FileNotFoundError(f"Required dataset not found: {BASE}")
    df = pd.read_csv(BASE, usecols=["date", "county", "daily_rainfall_in", "noaa_event_reported"])
    df["county"] = df["county"].astype(str).str.upper().str.strip()
    df = df.loc[df["county"] == "HARRIS"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date").sort_index()

@st.cache_data(show_spinner=False)
def read_extra_year(year):
    parts = []
    for _, _, _, key in SITES[1:]:
        path = WEATHER / f"harris__{key}__{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing downloaded weather file: {path}")
        df = pd.read_csv(path, usecols=["time", "precipitation"])
        df["date"] = pd.to_datetime(df["time"].str[:10], errors="raise").dt.date
        df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
        if df["precipitation"].isna().any():
            raise ValueError(f"Missing rainfall observations in {path}")
        parts.append((key, df.groupby("date")["precipitation"].sum() / 25.4))
    return dict(parts)

@st.cache_data(show_spinner=False)
def read_predictions():
    if not PREDICTIONS.exists():
        return None
    df = pd.read_csv(PREDICTIONS)
    needed = {"county", "date", "target_date", "forest_probability"}
    if not needed.issubset(df.columns):
        return None
    df = df.loc[df["county"].astype(str).str.upper().eq("HARRIS")].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")

@st.cache_data(show_spinner=False)
def load_harris_boundary():
    """Load a simplified county outline; cache it locally for future offline runs."""
    if BOUNDARY_CACHE.exists():
        geo = json.loads(BOUNDARY_CACHE.read_text(encoding="utf-8"))
        return shape(geo["geometry"] if geo.get("type") == "Feature" else geo)
    response = requests.get(BOUNDARY_URL, timeout=25)
    response.raise_for_status()
    counties = response.json()
    harris = next((f for f in counties["features"] if str(f.get("id", f.get("properties", {}).get("GEO_ID", ""))) == "48201"), None)
    if harris is None:
        raise ValueError("Harris County (FIPS 48201) was not found in the county boundary dataset")
    BOUNDARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BOUNDARY_CACHE.write_text(json.dumps(harris), encoding="utf-8")
    return shape(harris["geometry"])


def halfplane_polygon(a, b, bounds):
    """Huge polygon containing points closer to site a than site b.

    Uses locally scaled longitude to account approximately for latitude; the
    geometry is for display, not a physical flood prediction surface.
    """
    import math
    lat_scale = math.cos(math.radians((a[1] + b[1]) / 2))
    ax, ay = a[0] * lat_scale, a[1]
    bx, by = b[0] * lat_scale, b[1]
    dx, dy = bx - ax, by - ay
    mx, my = (ax + bx) / 2, (ay + by) / 2
    length = math.hypot(dx, dy)
    nx, ny = dx / length, dy / length
    tx, ty = -ny, nx
    extent = 100.0
    # Edge through midpoint, extending far along tangent; negative normal
    # selects the side nearer to a.
    xy = [
        (mx + tx * extent, my + ty * extent),
        (mx - tx * extent, my - ty * extent),
        (mx - tx * extent - nx * extent, my - ty * extent - ny * extent),
        (mx + tx * extent - nx * extent, my + ty * extent - ny * extent),
    ]
    return Polygon([(x / lat_scale, y) for x, y in xy])


@st.cache_data(show_spinner=False)
def make_zones():
    county = load_harris_boundary()
    zones = {}
    for i, (name, lat, lon, _) in enumerate(SITES):
        zone = county
        for j, (_, other_lat, other_lon, _) in enumerate(SITES):
            if i != j:
                zone = zone.intersection(halfplane_polygon((lon, lat), (other_lon, other_lat), county.bounds))
        if zone.is_empty:
            raise ValueError(f"Could not construct rainfall zone for {name}")
        zones[name] = mapping(zone)
    return zones, mapping(county.boundary)


try:
    base = read_base()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

min_date, max_date = base.index.min(), base.index.max()
def_date = date(2024, 5, 2) if date(2024, 5, 2) in base.index else max_date
if "playback_date" not in st.session_state:
    st.session_state.playback_date = def_date
if "playback_running" not in st.session_state:
    st.session_state.playback_running = False
if "playback_last_tick" not in st.session_state:
    st.session_state.playback_last_tick = time.monotonic()

@st.fragment(run_every="1s")
def dashboard():
    now = time.monotonic()
    if st.session_state.playback_running:
        elapsed_days = int((now - st.session_state.playback_last_tick) // 1.0)
        if elapsed_days > 0:
            new_date = min(max_date, st.session_state.playback_date + timedelta(days=elapsed_days))
            st.session_state.playback_date = new_date
            st.session_state.playback_last_tick += elapsed_days
            if new_date >= max_date:
                st.session_state.playback_running = False
    else:
        st.session_state.playback_last_tick = now

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.6rem 0 1.35rem 0;">
              <div style="display:flex; align-items:center; gap:0.9rem;">
                <div aria-hidden="true" style="font-size:2.8rem; line-height:1;">🌧️</div>
                <div style="min-width:0;">
                  <div style="font-size:2.5rem; font-weight:800; letter-spacing:0.02em; line-height:1.1;">SITR</div>
                  <div style="font-size:0.93rem; line-height:1.45; color:#e2e8f0;">Storm Intelligence &amp; Threat Recognition</div>
                </div>
              </div>
              <div style="border-top:1px solid rgba(148,163,184,0.28); margin-top:0.75rem; padding-top:0.55rem;">
                <div style="font-size:0.78rem; font-style:italic; color:#aab7ca; line-height:1.6;">
                  <span style="color:#7db6fa;">♫</span> Singing in the rain
                  <span style="font-style:normal; color:#8c98ac;"> — Jinsoul from LOONA</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Explore")
        chosen_input = st.date_input(
            "Historical weather date", value=st.session_state.playback_date,
            min_value=min_date, max_value=max_date,
            key=f"weather_date_{st.session_state.playback_date.isoformat()}"
        )
        if chosen_input != st.session_state.playback_date:
            st.session_state.playback_date = chosen_input
            st.session_state.playback_running = False
            st.session_state.playback_last_tick = time.monotonic()

        back_col, play_col, reset_col, next_col = st.columns(4)
        with back_col:
            if st.button("◀ Day", width="stretch", disabled=st.session_state.playback_date <= min_date):
                st.session_state.playback_date = max(min_date, st.session_state.playback_date - timedelta(days=1))
                st.session_state.playback_running = False
                st.session_state.playback_last_tick = time.monotonic()
        with play_col:
            if st.button("⏸ Pause" if st.session_state.playback_running else "▶ Play", width="stretch"):
                if st.session_state.playback_date >= max_date:
                    st.session_state.playback_date = date(2023, 1, 1)
                st.session_state.playback_running = not st.session_state.playback_running
                st.session_state.playback_last_tick = time.monotonic()
        with reset_col:
            if st.button("↺ Restart", width="stretch"):
                st.session_state.playback_date = date(2023, 1, 1)
                st.session_state.playback_running = False
                st.session_state.playback_last_tick = time.monotonic()
        with next_col:
            if st.button("Day ▶", width="stretch", disabled=st.session_state.playback_date >= max_date):
                st.session_state.playback_date = min(max_date, st.session_state.playback_date + timedelta(days=1))
                st.session_state.playback_running = False
                st.session_state.playback_last_tick = time.monotonic()
        st.caption("Playback speed: approximately 1 historical day per second. Pause to inspect any date.")
        st.markdown("**Rainfall legend (inches/day)**")
        st.markdown("🟢 Low: < 0.5  \n🟡 Moderate: 0.5–1.99  \n🟠 Heavy: 2.0–3.99  \n🔴 Very heavy: ≥ 4.0")
        st.markdown("**Flood report overlay**")
        st.markdown("🟣 Purple zones: NOAA flood event reported in Harris County that day")
        st.caption("No purple means no NOAA report in this dataset, not proof that no flooding occurred. Zones are illustrative.")
        st.caption("Zones are assigned to their nearest sampled location, not modeled neighborhood flood risk.")

    chosen = st.session_state.playback_date
    st.subheader(f"Historical weather · {chosen:%B %d, %Y}")
    if chosen not in base.index:
        st.warning("No county-day record available for this date.")
        st.stop()
    try:
        additional = read_extra_year(chosen.year)
        zones, boundary = make_zones()
    except (FileNotFoundError, ValueError, requests.RequestException, OSError, KeyError) as exc:
        st.error(f"Could not load map data: {exc}")
        st.info("For the geographic zones, install requests and shapely and allow one internet connection to download the county outline. The outline is then cached in data/harris_county_boundary.geojson.")
        st.stop()

    # NOAA reports are county-day labels, not individual neighborhood observations.
    flood_reported = bool(int(base.loc[chosen, "noaa_event_reported"]))

    original = float(base.loc[chosen, "daily_rainfall_in"])
    rows, features, label_rows = [], [], []
    for name, lat, lon, key in SITES:
        if key == "original":
            rain = original
        else:
            if chosen not in additional[key].index:
                st.error(f"No {name} rainfall data for {chosen}.")
                st.stop()
            rain = float(additional[key].loc[chosen])
        if pd.isna(rain):
            st.error(f"Missing rainfall for {name} on {chosen}.")
            st.stop()
        if rain < 0.5:
            color, level = [38, 166, 91, 170], "Low"
        elif rain < 2.0:
            color, level = [234, 179, 8, 170], "Moderate"
        elif rain < 4.0:
            color, level = [249, 115, 22, 170], "Heavy"
        else:
            color, level = [220, 38, 38, 170], "Very heavy"
        rows.append({"location": name, "rainfall": round(rain, 3), "level": level})
        label_rows.append({"location": name, "lat": lat, "lon": lon, "label": f"{name} zone", "rainfall": round(rain, 3)})
        features.append({"type": "Feature", "geometry": zones[name], "properties": {
            "location": name, "rainfall": round(rain, 3), "level": level,
            "color": [147, 51, 234, 205] if flood_reported else color,
            "flood_status": "NOAA flood reported in Harris County" if flood_reported else "No NOAA flood report in dataset",
        }})
    points = pd.DataFrame(rows)

    pred = read_predictions()
    score = None
    next_day = chosen + timedelta(days=1)
    if pred is not None and chosen in pred.index:
        match = pred.loc[chosen]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        if pd.to_datetime(match["target_date"]).date() == next_day:
            score = float(match["forest_probability"])

    if flood_reported:
        st.warning("🟣 NOAA flood event reported in Harris County on this date. All three illustrative zones turn purple because the event label is county-wide.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Maximum sampled rainfall", f"{points['rainfall'].max():.2f} in")
    c2.metric("Average sampled rainfall", f"{points['rainfall'].mean():.2f} in")
    c3.metric("Next-day flood-report risk score", f"{score:.1%}" if score is not None else "Not available")
    if score is not None:
        st.caption(f"For {next_day:%B %d, %Y} in Harris County: {score:.1%} model score for a next-day NOAA flood report. This is NOT a calibrated probability of flooding. Validation-selected alert cutoff: 37.9%. Score is {'above' if score >= 0.379205 else 'below'} the cutoff.")
    else:
        st.caption("A percentage score is available only when a saved historical prediction exists (primarily 2023–2025). The dashboard does not generate live forecasts.")

    zone_layer = pdk.Layer(
        "GeoJsonLayer", data={"type": "FeatureCollection", "features": features},
        filled=True, stroked=True, pickable=True,
        get_fill_color="properties.color",
        get_line_color=[20, 33, 50, 255], get_line_width=450,
        line_width_min_pixels=3,
    )
    outline_layer = pdk.Layer(
        "GeoJsonLayer", data={"type": "Feature", "geometry": boundary, "properties": {}},
        filled=False, stroked=True, pickable=False,
        get_line_color=[25, 35, 45, 255], get_line_width=300,
        line_width_min_pixels=2,
    )
    # Labels distinguish the three geographic zones even when they share a color.
    label_layer = pdk.Layer(
        "TextLayer", data=label_rows, get_position="[lon, lat]",
        get_text="label", get_size=19, size_min_pixels=13,
        get_color=[15, 23, 42, 255], get_angle=0,
        get_text_anchor="'middle'", get_alignment_baseline="'center'",
        pickable=False,
    )
    view = pdk.ViewState(latitude=29.82, longitude=-95.40, zoom=9.1, pitch=0)
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view, layers=[layer for layer in [zone_layer, outline_layer, label_layer] if layer is not None],
        tooltip={"html": "<b>{location} sampling zone</b><br/>Rainfall at sample: {rainfall} in<br/>Category: {level}<br/>{flood_status}",
                 "style": {"backgroundColor": "#14263c", "color": "white"}},
    ), width="stretch")
    st.caption("Purple zones indicate a county-wide NOAA flood report on the selected date; absence of purple does not establish no flooding. Three distinct zones—Houston, Katy, and Baytown—are separated by dark borders and labeled at their sample points. Pan, zoom, and hover over colored zones. Each area is assigned the rainfall of its nearest sampled location; rainfall is not measured at every point within the zone. Boundaries are illustrative partitions of Harris County, not official floodplain maps. Map tiles require internet access.")
    st.subheader("Weather observations")
    st.dataframe(points.rename(columns={"location": "Location", "rainfall": "Rainfall (in)", "level": "Rainfall category"}), hide_index=True, width="stretch")
    st.info("On non-report days, map colors represent historical ERA5 rainfall at three sampled locations. Purple indicates a county-wide NOAA flood report, not local flood probability. The percentage is a county-wide model score for a next-day NOAA flood report, not a verified chance of flooding or a public flood warning.")


with historical_tab:
    dashboard()


# Live forecast: separate current forecast from recent historical weather.
# The trained model estimates a next-day county-wide NOAA REPORT, not flooding itself.
LIVE_MODEL_PATH = DATA / "simplified_model_results" / "random_forest.joblib"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"
NWS_ALERT_URL = "https://api.weather.gov/alerts/active?zone=TXC201"
HOURLY_FIELDS = "precipitation,soil_moisture_0_to_7cm"


def fetch_open_meteo(url, params):
    response = requests.get(url, params=params, timeout=35)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "Open-Meteo API error"))
    return data


@st.cache_data(ttl=900, show_spinner=False)
def get_forecast():
    return fetch_open_meteo(FORECAST_URL, {
        "latitude": 29.7604, "longitude": -95.3698,
        "timezone": "America/Chicago", "past_days": 2, "forecast_days": 8,
        "hourly": HOURLY_FIELDS,
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "mm",
    })


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_history(today_iso):
    today = date.fromisoformat(today_iso)
    return fetch_open_meteo(HISTORY_URL, {
        "latitude": 29.7604, "longitude": -95.3698,
        "timezone": "America/Chicago",
        "start_date": (today - timedelta(days=12)).isoformat(),
        "end_date": (today - timedelta(days=2)).isoformat(),
        "hourly": "precipitation,soil_moisture_0_to_7cm", "wind_speed_unit": "mph",
        "precipitation_unit": "mm",
    })


@st.cache_data(ttl=900, show_spinner=False)
def get_nws_alerts():
    response = requests.get(NWS_ALERT_URL,
        headers={"User-Agent": "SITR educational flood research (contact: example@example.org)",
                 "Accept": "application/geo+json"}, timeout=20)
    response.raise_for_status()
    return response.json().get("features", [])


@st.cache_data(show_spinner=False)
def monthly_climatology():
    """Historical Harris County monthly medians; estimates, not observations."""
    csv_path = DATA / "extended_training_data.csv"
    if not csv_path.is_file():
        return {}
    columns = ["date", "county", "daily_rainfall_in", "max_hourly_rainfall_in", "avg_soil_moisture"]
    df = pd.read_csv(csv_path, usecols=columns)
    df = df.loc[df["county"].astype(str).str.upper().eq("HARRIS")].copy()
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
    for col in columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    result = {}
    for month, group in df.groupby("month"):
        result[int(month)] = {
            "rain": group["daily_rainfall_in"].median(),
            "max_hourly": group["max_hourly_rainfall_in"].median(),
            "soil": group["avg_soil_moisture"].median(),
        }
    return result


def hourly_daily_rows(payload, source="API"):
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {}
    df = pd.DataFrame({"time": times})
    df["day"] = pd.to_datetime(df["time"]).dt.date
    for field in ("precipitation", "soil_moisture_0_to_7cm"):
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(times):
            values = [None] * len(times)
        df[field] = pd.to_numeric(values, errors="coerce")
    result = {}
    for day, group in df.groupby("day"):
        rain = group["precipitation"]
        soil = group["soil_moisture_0_to_7cm"]
        result[day] = {
            "rain": float(rain.sum() / 25.4) if rain.notna().sum() >= 20 else None,
            "max_hourly": float(rain.max() / 25.4) if rain.notna().sum() >= 20 else None,
            "soil": float(soil.mean()) if soil.notna().any() else None,
            "source": source,
        }
    return result


def combine_daily_weather(forecast, history, today):
    """Prefer archive for past days; fill missing fields from forecast when possible."""
    forecast_days = hourly_daily_rows(forecast, "recent forecast")
    history_days = hourly_daily_rows(history, "historical archive") if history else {}
    days = dict(forecast_days)
    for day, archived in history_days.items():
        if day < today:
            fallback = forecast_days.get(day, {})
            days[day] = {field: archived.get(field) if archived.get(field) is not None else fallback.get(field)
                         for field in ("rain", "max_hourly", "soil")}
            days[day]["source"] = "historical archive + recent forecast" if any(
                archived.get(f) is None and fallback.get(f) is not None
                for f in ("rain", "max_hourly", "soil")) else "historical archive"
    return days


def model_input_for_target(days, target, climate):
    """Impute partial gaps, but never predict if rainfall history is entirely missing."""
    input_day = target - timedelta(days=1)
    required = [input_day - timedelta(days=i) for i in range(4)]
    values = {}
    estimated = []
    measured_rain_days = 0
    for day in required:
        record = days.get(day, {})
        values[day] = dict(record)
        if record.get("rain") is not None:
            measured_rain_days += 1
    if measured_rain_days < 2 or values[input_day].get("rain") is None:
        raise ValueError("Insufficient rainfall observations (need input-day rain and at least two of four days)")
    if values[input_day].get("max_hourly") is None:
        raise ValueError("Input-day maximum hourly rainfall unavailable")
    typical = climate.get(input_day.month, {})
    for day in required:
        for field in ("rain", "max_hourly", "soil"):
            if values[day].get(field) is not None:
                continue
            if field == "soil":
                # Use nearest earlier actual observation within seven days.
                candidates = [(d, r.get("soil")) for d, r in days.items()
                              if d <= day and (day - d).days <= 7 and r.get("soil") is not None]
                if candidates:
                    values[day][field] = max(candidates, key=lambda item: item[0])[1]
                    estimated.append(f"{day}: soil moisture carried from prior observation")
                    continue
            fallback = typical.get(field)
            if fallback is None or pd.isna(fallback):
                raise ValueError(f"Missing {field} for {day} and no Harris County monthly median available")
            values[day][field] = float(fallback)
            estimated.append(f"{day}: {field} replaced with historical monthly median")
    a, b, c, d = [values[day] for day in required]
    features = {
        "daily_rainfall_in": a["rain"],
        "max_hourly_rainfall_in": a["max_hourly"],
        "avg_soil_moisture": a["soil"],
        "rain_72h_prior_in": b["rain"] + c["rain"] + d["rain"],
    }
    return features, estimated, measured_rain_days


@st.cache_resource(show_spinner=False)
def get_saved_model():
    import joblib
    return joblib.load(LIVE_MODEL_PATH)


def predict_report_score(saved, values):
    model, columns = saved["model"], saved["features"]
    row = dict(values)
    for col in columns:
        if col.startswith("county_"):
            row[col] = int(col == "county_HARRIS")
    missing = [col for col in columns if col not in row]
    if missing:
        raise ValueError("Saved model requires unavailable features: " + ", ".join(missing))
    frame = pd.DataFrame([{col: row[col] for col in columns}], columns=columns)
    positive_index = list(model.classes_).index(1)
    return float(model.predict_proba(frame)[0, positive_index])


def live_dashboard():
    st.subheader("Houston · Live weather & previous 7 days")
    st.caption("Open-Meteo current forecast and historical archive · Harris County NWS alerts · Monthly-median fallback for partial gaps")
    if st.button("↻ Refresh live data", key="refresh_live_weather"):
        get_forecast.clear()
        get_recent_history.clear()
        get_nws_alerts.clear()

    today = datetime.now(ZoneInfo("America/Chicago")).date()
    try:
        forecast = get_forecast()
        daily = forecast.get("daily", {})
        dates = daily.get("time", [])
        if today.isoformat() not in dates:
            raise ValueError("Today's Houston forecast is unavailable")
        idx = dates.index(today.isoformat())
        def get_daily(field, index):
            values = daily.get(field, [])
            if index >= len(values) or values[index] is None:
                return None
            return float(values[index])

        rain = get_daily("precipitation_sum", idx)
        rain_pct = get_daily("precipitation_probability_max", idx)
        high = get_daily("temperature_2m_max", idx)
        low = get_daily("temperature_2m_min", idx)
        st.markdown(f"**Houston local date: {today:%B %d, %Y}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast rain today", f"{rain / 25.4:.2f} in" if rain is not None else "Unavailable")
        c2.metric("Chance of precipitation", f"{rain_pct:.0f}%" if rain_pct is not None else "Unavailable")
        c3.metric("High", f"{high:.0f}°F" if high is not None else "Unavailable")
        c4.metric("Low", f"{low:.0f}°F" if low is not None else "Unavailable")
        st.caption("Chance of precipitation is not a flood probability.")

        st.markdown("#### SITR AI · Previous seven days (excluding today)")
        st.caption("Retrospective next-day NOAA flood-report model scores. Each date is assessed using the weather from the preceding day; these are historical model estimates, not seven-day forecasts or actual flood probabilities.")
        if not LIVE_MODEL_PATH.is_file():
            st.warning("Simplified model missing at data/simplified_model_results/random_forest.joblib. Copy the trained model from the download package.")
        else:
            saved = get_saved_model()
            history = None
            retrieval_issues = []
            try:
                history = get_recent_history(today.isoformat())
            except (requests.RequestException, ValueError) as exc:
                retrieval_issues.append(f"Archive request failed: {exc}")
            days = combine_daily_weather(forecast, history, today)
            climate = monthly_climatology()
            records, problems, estimated_details = [], [], []
            for offset in range(7, 0, -1):
                target = today - timedelta(days=offset)
                try:
                    features, estimated, measured_count = model_input_for_target(days, target, climate)
                    score = predict_report_score(saved, features)
                    status = "Estimated inputs" if estimated else "Measured inputs"
                    records.append({
                        "Date": target.strftime("%a, %b %d"),
                        "AI flood-report score": round(score * 100, 1),
                        "Previous-day rain (in)": round(features["daily_rainfall_in"], 2),
                        "Previous 72h rain (in)": round(features["rain_72h_prior_in"], 2),
                        "Data status": status,
                    })
                    if estimated:
                        estimated_details.append(f"{target:%b %d}: " + "; ".join(estimated))
                except (ValueError, KeyError, IndexError) as exc:
                    records.append({
                        "Date": target.strftime("%a, %b %d"),
                        "AI flood-report score": None,
                        "Previous-day rain (in)": None,
                        "Previous 72h rain (in)": None,
                        "Data status": "Insufficient data",
                    })
                    problems.append(f"{target:%b %d}: {exc}")
            for issue in retrieval_issues:
                st.warning(issue + ". Recent forecast observations will be used where available.")
            table = pd.DataFrame(records)
            st.dataframe(table, hide_index=True, width="stretch",
                column_config={"AI flood-report score": st.column_config.NumberColumn(
                    "AI flood-report score (%)", format="%.1f%%")})
            chart_data = table.dropna(subset=["AI flood-report score"])
            if not chart_data.empty:
                st.line_chart(chart_data.set_index("Date")["AI flood-report score"],
                              y_label="Model score (%)", x_label="Past day")
            if estimated_details:
                with st.expander(f"Estimated-input details ({len(estimated_details)} days)"):
                    for issue in estimated_details:
                        st.write(issue)
            if problems:
                with st.expander(f"Weather/model diagnostics ({len(problems)} unavailable days)"):
                    for issue in problems:
                        st.write(issue)
            st.caption("**Data statuses:** Measured inputs = all model inputs from retrieved weather; Estimated inputs = some values substituted using prior observations or historical monthly medians; Insufficient data = prediction withheld. Substituted scores have not been validated.\n\n**Important:** These are uncalibrated Random Forest scores for a possible NOAA flood report in Harris County, NOT probabilities that flooding occurred. Historical forecast/reanalysis products can differ from the ERA5 training data. A low score does not establish flood-free conditions. This is not a warning system.")

    except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
        st.error(f"Live weather/AI forecast unavailable: {exc}")
        st.caption("The historical map is still available. Check network connectivity or expand the error details.")

    st.markdown("#### Official NWS flood alerts · Harris County")
    try:
        alerts = get_nws_alerts()
        flood_alerts = [f.get("properties", {}) for f in alerts
                        if any(word in str(f.get("properties", {}).get("event", "")).lower()
                               for word in ("flood", "hydrologic"))]
        if flood_alerts:
            for alert in flood_alerts:
                st.warning(f"**{alert.get('event', 'Flood alert')}** — "
                           f"{alert.get('headline') or alert.get('description', '')[:280]}")
                if alert.get("instruction"):
                    with st.expander("Official safety instructions"):
                        st.write(alert["instruction"])
        else:
            st.info("No active flood-related NWS alerts returned for Harris County. This does not guarantee no flooding.")
    except (requests.RequestException, ValueError) as exc:
        st.warning(f"NWS alert status unavailable: {exc}")
    st.markdown("[View official NWS alerts](https://www.weather.gov/)")
    st.caption("SITR is an educational research project, not an official flood warning system.")


with live_tab:
    live_dashboard()
