"""
microsite_prototype.py
======================
Pay-vs-Benchmark microsite prototype.

THROWAWAY CODE — for early CEO/customer feedback only.

Architecture (intentionally mirrors the eventual production design):

    UI (Streamlit)
         |
         v
    orchestrate()          <-- this layer talks to "APIs"
         |
         +--> benchmark_api()   <-- dummy: reads benchmark CSV, mimics real API
         +--> headcount_api()   <-- dummy: reads headcount CSV, mimics real API
         |
         v
    analyse()              <-- the PURE engine. Same as production.

Run:
    pip install streamlit pandas altair
    streamlit run microsite_prototype.py
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


# ====================================================================
# CONFIG
# ====================================================================
BENCHMARK_CSV = "benchmark_anon.csv"
HEADCOUNT_CSV = "headcount_anon.csv"

SHORT_CODE_TO_CATEGORY = {
    "IC":  "Individual Contributor",
    "M":   "Manager",
    "MGR": "Manager",
    "E":   "Executive",
    "D":   "Director",
    "S":   "Support",
    "I":   "Intern",
    "P":   "Professional",
}

FISCAL_YEAR_START_MONTH = 2
COMPONENTS = ["base", "commission", "bonus", "equity"]


# ====================================================================
# ENGINE  —  pure, no I/O, no API calls
# ====================================================================

@dataclass
class BenchmarkBand:
    base:       Optional[tuple]
    commission: Optional[tuple]
    bonus:      Optional[tuple]
    equity:     Optional[tuple]
    currency:   str


@dataclass
class AnalysisResult:
    out_of_band_rows: pd.DataFrame
    quarterly_aggregates: pd.DataFrame
    summary: dict


def _classify(actual, band):
    if band is None or pd.isna(actual):
        return None, None, None
    low, _mid, high = band
    if actual < low:
        return "below", (low - actual) / low * 100, low - actual
    if actual > high:
        return "above", (actual - high) / high * 100, actual - high
    return "within", None, None


def _fiscal_quarter_freq(fiscal_year_start_month):
    if not 1 <= fiscal_year_start_month <= 12:
        raise ValueError(f"fiscal_year_start_month must be 1..12, got {fiscal_year_start_month}")
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    end_month_index = (fiscal_year_start_month - 2) % 12
    return f"Q-{months[end_month_index]}"


def analyse(band, headcount, date_from, date_to, fiscal_year_start_month):
    mask = (headcount["start_date"] >= pd.Timestamp(date_from)) & \
           (headcount["start_date"] <= pd.Timestamp(date_to))
    in_window = headcount.loc[mask].copy()

    records = []
    for _, row in in_window.iterrows():
        for comp in COMPONENTS:
            actual = row.get(f"actual_{comp}")
            comp_band = getattr(band, comp)
            position, var_pct, var_amt = _classify(actual, comp_band)
            records.append({
                "headcount_id":    row.get("headcount_id"),
                "employee_id":     row.get("employee_id"),
                "name":            row["name"],
                "start_date":      row["start_date"],
                "hiring_status":   row.get("hiring_status"),
                "component":       comp,
                "actual":          actual,
                "band_low":        comp_band[0] if comp_band else None,
                "band_high":       comp_band[2] if comp_band else None,
                "position":        position,
                "variance_pct":    var_pct,
                "variance_amount": var_amt,
            })
    classified = pd.DataFrame(records)

    out_of_band_rows = classified[classified["position"].isin(["below", "above"])].copy()

    freq = _fiscal_quarter_freq(fiscal_year_start_month)
    classified["fiscal_quarter"] = classified["start_date"].dt.to_period(freq).astype(str)

    def _quarter_agg(group):
        evaluable = group[group["position"].notna()]
        flagged   = evaluable[evaluable["position"].isin(["below", "above"])]
        return pd.Series({
            "n_matched":           len(evaluable),
            "n_out_of_band":       len(flagged),
            "share_out_of_band":   (len(flagged) / len(evaluable)) if len(evaluable) else np.nan,
            "avg_variance_amount": flagged["variance_amount"].mean() if len(flagged) else np.nan,
        })

    quarterly_aggregates = (
        classified.groupby(["fiscal_quarter", "component"], dropna=False)
                  .apply(_quarter_agg, include_groups=False)
                  .reset_index()
    )

    summary = {
        "n_employees_in_window":   int(in_window["headcount_id"].nunique()) if "headcount_id" in in_window.columns else len(in_window),
        "n_flagged_rows":          int(len(out_of_band_rows)),
        "components_with_band":    [c for c in COMPONENTS if getattr(band, c) is not None],
        "currency":                band.currency,
        "fiscal_year_start_month": fiscal_year_start_month,
        "date_from":               date_from,
        "date_to":                 date_to,
    }
    return AnalysisResult(out_of_band_rows.reset_index(drop=True),
                          quarterly_aggregates, summary)


# ====================================================================
# DUMMY APIs  —  read from CSV, mimic real endpoints
# ====================================================================

@st.cache_data
def _load_benchmark_raw():
    return pd.read_csv(BENCHMARK_CSV)

@st.cache_data
def _load_headcount_raw():
    df = pd.read_csv(HEADCOUNT_CSV)
    for col in ["Actual salary", "Actual commission", "Actual bonus amount",
                "Actual equity", "Actual bonus percentage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).replace({"nan": np.nan}),
                errors="coerce"
            )
    df["Actual start date"] = pd.to_datetime(df["Actual start date"], format="%m-%d-%Y", errors="coerce")
    return df


def benchmark_api(department, job_category, job_level, job_role, location):
    df = _load_benchmark_raw()
    mask = (
        (df["Department"]   == department)   &
        (df["Job Category"] == job_category) &
        (df["Job Level"]    == job_level)    &
        (df["Job Role"]     == job_role)     &
        (df["Location"]     == location)
    )
    matches = df[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]

    def _band(prefix):
        lo, mid, hi = row[f"{prefix} Low"], row[f"{prefix} Mid"], row[f"{prefix} High"]
        if pd.isna(lo) and pd.isna(mid) and pd.isna(hi):
            return None
        if pd.isna(lo) or pd.isna(hi):
            return None
        return (lo, mid if not pd.isna(mid) else (lo + hi) / 2, hi)

    return BenchmarkBand(
        base       = _band("Base"),
        commission = _band("Commission"),
        bonus      = _band("Bonus"),
        equity     = _band("Equity"),
        currency   = row["Currency"],
    )


def headcount_api(department, job_role, short_code, location, date_from, date_to):
    df = _load_headcount_raw()
    mask = (
        (df["Department"] == department) &
        (df["Job role"]   == job_role)   &
        (df["Job level"]  == short_code) &
        (df["Location"]   == location)   &
        (df["Employee type"] != "Intern") &
        (df["Actual start date"] >= pd.Timestamp(date_from)) &
        (df["Actual start date"] <= pd.Timestamp(date_to))
    )
    return df[mask].copy()


# ====================================================================
# ORCHESTRATOR
# ====================================================================

def _parse_short_code(code):
    if pd.isna(code):
        raise ValueError("Short code is missing")
    m = re.match(r"^([A-Z]+)(\d+)$", str(code).strip())
    if not m:
        raise ValueError(f"Short code '{code}' does not match expected pattern like 'IC5'")
    prefix, level = m.group(1), int(m.group(2))
    if prefix not in SHORT_CODE_TO_CATEGORY:
        raise ValueError(f"Unknown short code prefix '{prefix}' in '{code}'.")
    return SHORT_CODE_TO_CATEGORY[prefix], level


def orchestrate(department, job_role, short_code, location, date_from, date_to):
    job_category, job_level = _parse_short_code(short_code)
    band = benchmark_api(department, job_category, job_level, job_role, location)
    headcount_rows = headcount_api(department, job_role, short_code, location, date_from, date_to)

    if band is None:
        return None, headcount_rows, "No benchmark found for this combination."

    if not headcount_rows.empty:
        headcount_rows = headcount_rows[headcount_rows["Actual pay currency"] == band.currency]

    engine_rows = headcount_rows.rename(columns={
        "Headcount ID":          "headcount_id",
        "Employee Name":         "name",
        "Actual start date":     "start_date",
        "Hiring status":         "hiring_status",
        "Actual salary":         "actual_base",
        "Actual commission":     "actual_commission",
        "Actual bonus amount":   "actual_bonus",
        "Actual equity":         "actual_equity",
    })
    engine_rows["employee_id"] = engine_rows["headcount_id"]

    result = analyse(band, engine_rows, date_from, date_to, FISCAL_YEAR_START_MONTH)
    return band, headcount_rows, result


# ====================================================================
# UI HELPERS
# ====================================================================

def _fmt_money(value, currency):
    if pd.isna(value):
        return "—"
    return f"{currency} {value:,.0f}"

def _position_pill(position):
    if position == "below":
        color, bg = "#9a1f23", "#fce8e8"
    elif position == "above":
        color, bg = "#7b4e00", "#fff4d6"
    else:
        color, bg = "#1f5e3a", "#e6f4ec"
    return f'<span style="color:{color};background:{bg};padding:2px 8px;border-radius:10px;font-size:0.85em;font-weight:600">{position}</span>'


# ====================================================================
# UI
# ====================================================================

st.set_page_config(page_title="Pay-vs-Benchmark", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main-header { padding: 1.25rem 0 0.5rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem; }
.main-header h1 { margin: 0; font-size: 1.6rem; color: #111827; }
.main-header .tagline { color: #6b7280; font-size: 0.95rem; margin-top: 0.25rem; }
.kpi { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.85rem 1rem; }
.kpi .label { color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi .value { color: #111827; font-size: 1.55rem; font-weight: 600; margin-top: 0.2rem; }
.kpi .value.flagged { color: #9a1f23; }
.section-title { font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 1.75rem; margin-bottom: 0.6rem; }
.subtle { color: #6b7280; font-size: 0.9rem; }
table.dataframe { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.dataframe th { background: #f3f4f6; text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e7eb; }
table.dataframe td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
table.wide-flagged td { min-width: 130px; }
table.wide-flagged th { white-space: nowrap; }
table.wide-quarter td { padding: 0.4rem; min-width: 110px; }
table.wide-quarter th { white-space: nowrap; }
table.wide-quarter td:first-child, table.wide-quarter td:nth-child(2) {
  vertical-align: middle; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>Pay-vs-Benchmark</h1>
  <div class="tagline">Where actual pay falls outside the bands you adopted — by role, level, and location.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
h = _load_headcount_raw()
h_active = h[h["Employee type"] != "Intern"].copy()

with st.sidebar:
    st.markdown("### Filter")
    departments = sorted(h_active["Department"].dropna().unique())
    department = st.selectbox("Department", departments)

    job_roles = sorted(h_active[h_active["Department"] == department]["Job role"].dropna().unique())
    job_role = st.selectbox("Job Role", job_roles)

    short_codes = sorted(
        h_active[(h_active["Department"] == department) &
                 (h_active["Job role"] == job_role)]["Job level"].dropna().unique()
    )
    short_code = st.selectbox("Job Level", short_codes)

    locations = sorted(
        h_active[(h_active["Department"] == department) &
                 (h_active["Job role"] == job_role) &
                 (h_active["Job level"] == short_code)]["Location"].dropna().unique()
    )
    location = st.selectbox("Location", locations)

    st.markdown("")
    min_d = h_active["Actual start date"].min().date()
    max_d = h_active["Actual start date"].max().date()
    date_from = st.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
    date_to   = st.date_input("To",   value=max_d, min_value=min_d, max_value=max_d)

    st.markdown("")
    run = st.button("Run analysis", type="primary", use_container_width=True)


if not run:
    st.markdown('<div class="subtle">Pick filters in the sidebar and click <b>Run analysis</b>.</div>',
                unsafe_allow_html=True)
    st.stop()

try:
    band, raw_rows, result = orchestrate(department, job_role, short_code, location, date_from, date_to)
except ValueError as e:
    st.error(f"Orchestrator error: {e}")
    st.stop()

st.markdown(
    f'<div class="subtle"><b>{department}</b> · <b>{job_role}</b> · '
    f'<b>{short_code}</b> · <b>{location}</b> · '
    f'{date_from:%b %Y} → {date_to:%b %Y}</div>',
    unsafe_allow_html=True
)
st.markdown("")

if band is None:
    st.warning(
        f"**No benchmark band found** for this combination. "
        f"There are {len(raw_rows)} matching hires in this date window, but the job catalog doesn't yet have a band for this role at this location."
    )
    st.stop()

if isinstance(result, str):
    st.warning(result)
    st.stop()

# KPIs
n_hires    = result.summary["n_employees_in_window"]
n_flagged  = result.summary["n_flagged_rows"]
n_below    = (result.out_of_band_rows["position"] == "below").sum() if not result.out_of_band_rows.empty else 0
n_above    = (result.out_of_band_rows["position"] == "above").sum() if not result.out_of_band_rows.empty else 0
currency   = result.summary["currency"]

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><div class="label">Hires in window</div><div class="value">{n_hires}</div></div>',
            unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="label">Flagged (out of band)</div><div class="value flagged">{n_flagged}</div></div>',
            unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="label">Below band</div><div class="value">{n_below}</div></div>',
            unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="label">Above band</div><div class="value">{n_above}</div></div>',
            unsafe_allow_html=True)

# Benchmark band
st.markdown('<div class="section-title">Benchmark band</div>', unsafe_allow_html=True)
band_rows = []
for c in COMPONENTS:
    bb = getattr(band, c)
    band_rows.append({
        "Component": c.capitalize(),
        "Low":  _fmt_money(bb[0] if bb else None, currency),
        "Mid":  _fmt_money(bb[1] if bb else None, currency),
        "High": _fmt_money(bb[2] if bb else None, currency),
        "Status": "Active" if bb else "No band published",
    })
st.dataframe(pd.DataFrame(band_rows), hide_index=True, use_container_width=True)

# Flagged hires (WIDE layout: one row per hire, four component columns)
# The engine produces long format (one row per employee × component) because
# that shape is right for aggregation. The UI pivots it back to wide because
# wide is right for human reading — eye scans across a hire to see its full
# pattern across all four components.
flagged_employee_ids = result.out_of_band_rows["headcount_id"].unique() if not result.out_of_band_rows.empty else []

st.markdown(
    f'<div class="section-title">Flagged hires <span class="subtle">({len(flagged_employee_ids)} of {n_hires})</span></div>',
    unsafe_allow_html=True
)

if len(flagged_employee_ids) == 0:
    st.success("No hires fell outside the band in this window.")
else:
    # We need ALL component rows for the flagged employees (not just the flagged
    # ones), so the wide row can show every component's status. Re-derive from
    # the engine's full classified set by re-running classification on the
    # original headcount data — but the engine doesn't return the in-band rows.
    # Easiest path: pivot from the long classified set the engine produced,
    # combined with the same-shape data we can reconstruct from raw_rows + band.

    def _build_component_cell(actual, band, position, var_pct, var_amt, currency):
        """Compact stacked cell: actual amount, coloured pill, variance (if flagged)."""
        if pd.isna(actual) and band is None:
            return '<div style="color:#9ca3af;font-size:0.85em">—</div>'
        if pd.isna(actual):
            return f'<div style="color:#9ca3af">no actual<br><span style="font-size:0.75em">band: {_fmt_money(band[0], currency)}–{_fmt_money(band[2], currency)}</span></div>'
        amt = _fmt_money(actual, currency)
        if position is None:
            return f'<div>{amt}<br><span style="color:#9ca3af;font-size:0.78em">no band</span></div>'
        if position == "within":
            return f'<div>{amt}<br><span style="color:#1f5e3a;background:#e6f4ec;padding:1px 6px;border-radius:8px;font-size:0.72em;font-weight:600">within</span></div>'
        color, bg = ("#9a1f23", "#fce8e8") if position == "below" else ("#7b4e00", "#fff4d6")
        variance_line = f'<span style="color:{color};font-size:0.78em">{_fmt_money(var_amt, currency)} · {var_pct:.1f}%</span>'
        return (f'<div>{amt}<br>'
                f'<span style="color:{color};background:{bg};padding:1px 6px;border-radius:8px;font-size:0.72em;font-weight:600">{position}</span> '
                f'{variance_line}</div>')

    # Build wide rows. For each flagged hire, look up each component's classification.
    long_df = result.out_of_band_rows  # long, but only the flagged component rows
    # We also need the within-band component rows for context. The engine
    # discards them, but we can reconstruct from `raw_rows` + the band.
    raw_lookup = raw_rows.set_index("Headcount ID")

    wide_rows = []
    for hid in flagged_employee_ids:
        if hid not in raw_lookup.index:
            continue
        rr = raw_lookup.loc[hid]
        if isinstance(rr, pd.DataFrame):  # duplicate IDs shouldn't happen but defend
            rr = rr.iloc[0]
        actuals = {
            "base":       rr.get("Actual salary"),
            "commission": rr.get("Actual commission"),
            "bonus":      rr.get("Actual bonus amount"),
            "equity":     rr.get("Actual equity"),
        }
        # Classify each component to get position/variance for the cell
        comp_cells = {}
        for comp in COMPONENTS:
            b = getattr(band, comp)
            pos, vpct, vamt = _classify(actuals[comp], b)
            comp_cells[comp] = _build_component_cell(actuals[comp], b, pos, vpct, vamt, currency)

        wide_rows.append({
            "Hire":       f"{rr['Employee Name']}<br><span style='color:#6b7280;font-size:0.78em'>{hid}</span>",
            "Status":     rr["Hiring status"],
            "Start date": pd.to_datetime(rr["Actual start date"]).strftime("%d %b %Y"),
            "Base":       comp_cells["base"],
            "Commission": comp_cells["commission"],
            "Bonus":      comp_cells["bonus"],
            "Equity":     comp_cells["equity"],
        })

    wide_df = pd.DataFrame(wide_rows)
    # Wrap in a horizontally-scrollable div so the wide layout doesn't break
    # narrow viewports — accepted tradeoff for cell compactness.
    html = wide_df.to_html(escape=False, index=False, classes="dataframe wide-flagged")
    st.markdown(
        f'<div style="overflow-x:auto">{html}</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

# Quarterly trend
st.markdown('<div class="section-title">Quarterly trend</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Share of hires falling outside the band, per fiscal quarter. Each cell shows '
    '% flagged with the average variance underneath; redder cells flag a higher share off-band.</div>',
    unsafe_allow_html=True
)
st.markdown("")

qa = result.quarterly_aggregates.copy()

if qa.empty or (qa["n_matched"].sum() == 0):
    st.info("No quarterly data in this window.")
else:
    # Bar chart first (visual scan), wide heat-table below (numbers).
    qa_chart = qa[qa["n_matched"] > 0].copy()
    qa_chart["share_pct"] = (qa_chart["share_out_of_band"] * 100).fillna(0)
    qa_chart["component"] = qa_chart["component"].str.capitalize()

    chart = alt.Chart(qa_chart).mark_bar().encode(
        x=alt.X("fiscal_quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share_pct:Q",      title="% flagged"),
        color=alt.Color("component:N", legend=None,
                       scale=alt.Scale(scheme="set2")),
        column=alt.Column("component:N", title=None, header=alt.Header(labelFontSize=12)),
        tooltip=[
            alt.Tooltip("fiscal_quarter:N",  title="Quarter"),
            alt.Tooltip("component:N",       title="Component"),
            alt.Tooltip("n_matched:Q",       title="Hires with band"),
            alt.Tooltip("n_out_of_band:Q",   title="Flagged"),
            alt.Tooltip("share_pct:Q",       title="% flagged", format=".1f"),
            alt.Tooltip("avg_variance_amount:Q", title="Avg variance", format=",.0f"),
        ],
    ).properties(height=220, width=160)
    st.altair_chart(chart, use_container_width=False)

    st.markdown("")

    # Wide heat-table: one row per quarter, four component columns.
    # Heat intensity: cell background reddens as % flagged increases (0% white, 100% saturated red).
    def _heat_cell(pct, n_matched, avg_var, currency):
        """Compact heat-coloured cell: % flagged top, avg variance underneath."""
        if n_matched == 0 or pd.isna(n_matched):
            return '<div style="color:#9ca3af;font-size:0.85em;text-align:center">—</div>'
        if pd.isna(pct):
            pct = 0
        # Red heat: lerp from white (0%) to deep red (100%). Use a perceptual mid-point.
        # We cap intensity at 100% — anything 50%+ is already vivid.
        intensity = min(pct / 100.0, 1.0)
        # Background: from #ffffff -> #b91c1c (matching the "below" pill family)
        r = int(255 - (255 - 185) * intensity)
        g = int(255 - (255 - 28)  * intensity)
        b = int(255 - (255 - 28)  * intensity)
        bg = f"rgb({r},{g},{b})"
        # Text colour: dark on light, white on dark (transition around 50%)
        text_color = "#111827" if intensity < 0.5 else "#ffffff"
        muted_color = "#6b7280" if intensity < 0.5 else "rgba(255,255,255,0.85)"

        pct_line = f'<div style="font-size:1.0em;font-weight:600;color:{text_color}">{pct:.0f}%</div>'
        if pd.isna(avg_var) or avg_var == 0:
            var_line = f'<div style="font-size:0.72em;color:{muted_color}">—</div>'
        else:
            var_line = f'<div style="font-size:0.72em;color:{muted_color}">avg {_fmt_money(avg_var, currency)}</div>'
        return f'<div style="background:{bg};padding:0.5rem;border-radius:4px;text-align:center">{pct_line}{var_line}</div>'

    # Pivot: rows = quarter, columns = component
    quarters = sorted(qa["fiscal_quarter"].dropna().unique())
    wide_q_rows = []
    for q in quarters:
        q_df = qa[qa["fiscal_quarter"] == q]
        # Total hires in this quarter (across all evaluable components — use the
        # max n_matched as the per-quarter hire count; n_matched can differ by
        # component because not every component has a band, but the hire count
        # itself is shared across components for the same hires).
        n_hires_q = int(q_df["n_matched"].max()) if not q_df.empty else 0
        row = {"Quarter": q, "Hires with band": n_hires_q}
        for comp in COMPONENTS:
            cdf = q_df[q_df["component"] == comp]
            if cdf.empty:
                row[comp.capitalize()] = _heat_cell(np.nan, 0, np.nan, currency)
                continue
            r0 = cdf.iloc[0]
            pct = (r0["share_out_of_band"] * 100) if pd.notna(r0["share_out_of_band"]) else np.nan
            row[comp.capitalize()] = _heat_cell(pct, r0["n_matched"], r0["avg_variance_amount"], currency)
        wide_q_rows.append(row)

    wide_q_df = pd.DataFrame(wide_q_rows)
    st.markdown(
        f'<div style="overflow-x:auto">{wide_q_df.to_html(escape=False, index=False, classes="dataframe wide-quarter")}</div>',
        unsafe_allow_html=True
    )

st.markdown("")
st.markdown(
    '<div class="subtle" style="margin-top:2rem;padding-top:1rem;border-top:1px solid #e5e7eb">'
    'Prototype build · dummy data · for early feedback only'
    '</div>',
    unsafe_allow_html=True
)
