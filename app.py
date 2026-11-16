"""
app.py  -  LifeStat: Personal Lifestyle Statistical Analyzer
------------------------------------------------------------
Run locally with:   streamlit run app.py

This file contains the user interface only. The mathematics lives in
stats_engine.py, data checking in data_utils.py and reports in report_builder.py.
"""

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_utils as du
import report_builder as rb
import stats_engine as se

st.set_page_config(
    page_title="LifeStat - Personal Lifestyle Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colours and styling
# ---------------------------------------------------------------------------
GREEN_DARK = "#1B5E3B"
GREEN = "#2E8B57"
GREEN_LIGHT = "#A8D5BA"
SAGE = "#EAF3EC"
CREAM = "#F7F5EF"
GOLD = "#E9B44C"
CLAY = "#B5543C"
TEXT = "#22302A"
GRID = "#E3EAE5"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background-color: #F7F5EF; }
[data-testid="stSidebar"] { background-color: #EAF3EC; border-right: 1px solid #D5E5DA; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #1B5E3B 0%, #2E8B57 100%);
    color: #FFFFFF; padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1.1rem;
}
.hero h1 { color: #FFFFFF; font-size: 1.8rem; margin: 0 0 .25rem 0; padding: 0; }
.hero p { color: #E4F2EA; margin: 0; font-size: 1rem; }

.metric-card {
    background: #FFFFFF; border-radius: 14px; padding: .9rem 1rem; margin-bottom: .7rem;
    border-left: 5px solid #2E8B57; box-shadow: 0 1px 4px rgba(27, 94, 59, 0.10);
}
.metric-label { color: #5B6B62; font-size: .82rem; font-weight: 500; }
.metric-value { color: #1B5E3B; font-size: 1.7rem; font-weight: 700; line-height: 1.25; }
.metric-sub { color: #7A887F; font-size: .76rem; }

.insight {
    background: #FFFFFF; border-left: 5px solid #E9B44C; border-radius: 10px;
    padding: .85rem 1.1rem; margin: .6rem 0 1rem 0; color: #22302A;
}
.note {
    background: #EAF3EC; border-radius: 10px; padding: .75rem 1rem; margin: .5rem 0 1rem 0;
    color: #22302A; font-size: .93rem;
}
.brand { font-size: 1.5rem; font-weight: 700; color: #1B5E3B; margin-bottom: 0; }
.brand-sub { color: #5B6B62; font-size: .82rem; margin-bottom: .8rem; }

@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
    .hero h1 { font-size: 1.4rem; }
    .metric-value { font-size: 1.4rem; }
}
</style>
"""

CONDITIONS = {">=": "at least", "<=": "at most", ">": "more than", "<": "less than"}
DEFAULT_THRESHOLD = {
    "sleep_hours": 7.0,
    "exercise_minutes": 30.0,
    "water_litres": 2.0,
    "screen_hours": 6.0,
    "study_hours": 4.0,
    "energy": 7.0,
}
STEP = {
    "sleep_hours": 0.5,
    "exercise_minutes": 5.0,
    "water_litres": 0.1,
    "screen_hours": 0.5,
    "study_hours": 0.5,
    "energy": 1.0,
}
DEFAULT_RANGE = {"sleep_hours": (7.0, 9.0)}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fmt(value, digits=2):
    """Format a number for display; show a dash if it is missing."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def insight(html_text):
    st.markdown(f'<div class="insight">{html_text}</div>', unsafe_allow_html=True)


def note(html_text):
    st.markdown(f'<div class="note">{html_text}</div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def style_fig(fig, title=None, height=340):
    """Apply the LifeStat look to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show_fig(fig):
    st.plotly_chart(fig)


def init_state():
    """Start every new visit with the built-in sample data."""
    if "df" not in st.session_state:
        st.session_state.df = du.load_sample()
        st.session_state.source = "sample"
    if "flash" not in st.session_state:
        st.session_state.flash = None


def set_flash(kind, message):
    st.session_state.flash = (kind, message)


def show_flash():
    flash = st.session_state.get("flash")
    if flash:
        kind, message = flash
        {"success": st.success, "warning": st.warning, "info": st.info}.get(kind, st.info)(message)
        st.session_state.flash = None


def data_banner():
    source = st.session_state.source
    if source == "sample":
        st.info(
            "📌 You are viewing the built-in **sample data** (illustrative, not real). "
            "Add your own records or upload a CSV file to analyse your own habits."
        )
    elif source == "mixed":
        st.warning("📌 Your data currently includes built-in **sample records** mixed with your own.")


def empty_state(message="No records yet."):
    st.markdown(
        f'<div class="note">🌱 <b>{message}</b><br>Use <b>Daily Entry</b> to add a record, '
        f"<b>Upload Data</b> to import a CSV file, or reset to the sample data in <b>My Records</b>.</div>",
        unsafe_allow_html=True,
    )


def require_records(minimum=1):
    """Return True when there is enough data; otherwise show a helpful message."""
    count = len(st.session_state.df)
    if count == 0:
        empty_state("There is no data to analyse.")
        return False
    if count < minimum:
        st.info(
            f"🌱 This analysis needs at least {minimum} records. You currently have {count}. "
            "Add a few more days and it will appear here."
        )
        return False
    return True


def store_records(new, replace_all):
    """Save new records into the session, updating the data-source label."""
    current = st.session_state.df
    source = st.session_state.source
    if replace_all or len(current) == 0:
        combined, replaced = new.sort_values("date").reset_index(drop=True), 0
        st.session_state.source = "user"
    else:
        combined, replaced = du.combine_records(current, new)
        st.session_state.source = "mixed" if source in ("sample", "mixed") else "user"
    st.session_state.df = combined
    return replaced


def numeric_choice(label, key, default_col):
    return st.selectbox(
        label,
        du.NUMERIC_COLUMNS,
        index=du.NUMERIC_COLUMNS.index(default_col),
        format_func=lambda c: du.LABELS[c],
        key=key,
    )


# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
def page_overview():
    page_header("Your Lifestyle at a Glance", "A friendly summary of your sleep, activity, hydration and energy.")
    show_flash()
    data_banner()
    df = st.session_state.df
    if len(df) == 0:
        empty_state()
        return

    first = df["date"].min().strftime("%d %b %Y")
    last = df["date"].max().strftime("%d %b %Y")
    st.caption(f"Records from {first} to {last}")

    row1 = st.columns(4)
    with row1[0]:
        metric_card("Average sleep", f"{df['sleep_hours'].mean():.1f} h", "per night")
    with row1[1]:
        metric_card("Average exercise", f"{df['exercise_minutes'].mean():.0f} min", "per day")
    with row1[2]:
        metric_card("Average water intake", f"{df['water_litres'].mean():.1f} L", "per day")
    with row1[3]:
        metric_card("Average screen time", f"{df['screen_hours'].mean():.1f} h", "per day")
    row2 = st.columns(3)
    with row2[0]:
        metric_card("Average study/work", f"{df['study_hours'].mean():.1f} h", "per day")
    with row2[1]:
        metric_card("Average energy", f"{df['energy'].mean():.1f} / 10", "daily rating")
    with row2[2]:
        metric_card("Total recorded days", f"{len(df)}", "days of data")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_hours"], mode="lines+markers", name="Sleep",
                                 line=dict(color=GREEN_LIGHT, width=2), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_hours"].rolling(7, min_periods=1).mean(),
                                 mode="lines", name="7-day average", line=dict(color=GREEN_DARK, width=3)))
        fig.add_hline(y=7, line_dash="dot", line_color=GOLD, annotation_text="7 h")
        fig.update_yaxes(title="Hours")
        show_fig(style_fig(fig, "Sleep duration trend"))
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["energy"], mode="lines+markers", name="Energy",
                                 line=dict(color=GREEN, width=2), marker=dict(size=5)))
        fig.update_yaxes(title="Rating", range=[0, 10.5])
        show_fig(style_fig(fig, "Daily energy trend"))

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=df["date"], y=df["exercise_minutes"], marker_color=GREEN, name="Exercise"))
        fig.update_yaxes(title="Minutes")
        show_fig(style_fig(fig, "Exercise activity"))
    with right:
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_day = df.groupby(df["date"].dt.dayofweek)["energy"].mean().reindex(range(7))
        fig = go.Figure(go.Bar(x=names, y=by_day.values, marker_color=GREEN_LIGHT, name="Energy"))
        fig.update_yaxes(title="Average energy", range=[0, 10])
        show_fig(style_fig(fig, "Average energy by day of the week"))

    moods = df.loc[df["mood"] != du.NOT_RECORDED, "mood"].value_counts()
    if len(moods) > 0:
        fig = go.Figure(go.Bar(x=moods.index.tolist(), y=moods.values, marker_color=GREEN, name="Days"))
        fig.update_yaxes(title="Days")
        show_fig(style_fig(fig, "Mood overview", height=300))


# ---------------------------------------------------------------------------
# PAGE: Daily Entry
# ---------------------------------------------------------------------------
def page_daily_entry():
    page_header("Daily Entry", "Record how your day went. Add as many days as you like.")
    show_flash()
    data_banner()

    with st.form("daily_entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            entry_date = st.date_input("Date", value=date.today())
            sleep = st.number_input("Sleep duration (hours)", min_value=0.0, max_value=24.0,
                                    value=7.0, step=0.25)
            exercise = st.number_input("Exercise duration (minutes)", min_value=0, max_value=1440,
                                       value=30, step=5)
            water = st.number_input("Water intake (litres)", min_value=0.0, max_value=10.0,
                                    value=2.0, step=0.1)
        with c2:
            screen = st.number_input("Screen time (hours)", min_value=0.0, max_value=24.0,
                                     value=5.0, step=0.5)
            study = st.number_input("Study/work hours", min_value=0.0, max_value=24.0,
                                    value=4.0, step=0.5)
            energy = st.slider("Energy rating (1-10)", min_value=1, max_value=10, value=6)
            mood = st.selectbox("Mood", du.MOOD_OPTIONS, index=1)
            quality = st.selectbox("Sleep quality", du.SLEEP_QUALITY_OPTIONS, index=2)

        replace_sample = True
        if st.session_state.source == "sample":
            replace_sample = st.checkbox(
                "Replace the sample data with my own records (recommended)", value=True
            )
        submitted = st.form_submit_button("➕ Add record")

    if submitted:
        values = {
            "date": pd.Timestamp(entry_date),
            "sleep_hours": sleep,
            "exercise_minutes": exercise,
            "water_litres": water,
            "screen_hours": screen,
            "study_hours": study,
            "energy": energy,
            "mood": mood,
            "sleep_quality": quality,
        }
        problems = du.row_issues(values)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            new = du.make_record_frame(values)
            replace_all = st.session_state.source == "sample" and replace_sample
            replaced = store_records(new, replace_all)
            message = f"Record for {entry_date:%d %b %Y} saved for this session."
            if replaced:
                message += " An earlier record for the same date was replaced."
            set_flash("success", message)
            st.rerun()

    st.markdown("#### Most recent records")
    df = st.session_state.df
    if len(df) == 0:
        empty_state()
    else:
        st.dataframe(du.display_frame(df.tail(7).iloc[::-1]), hide_index=True)
    note("💡 Records live in your current browser session only. Use <b>My Records</b> to download a "
         "CSV copy so you can upload it again next time.")


# ---------------------------------------------------------------------------
# PAGE: Upload Data
# ---------------------------------------------------------------------------
def read_uploaded_csv(uploaded):
    raw_bytes = uploaded.getvalue()
    if len(raw_bytes) > du.MAX_UPLOAD_BYTES:
        return None, "The file is larger than 5 MB. Please upload a smaller file."
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding), None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The file is empty."
        except Exception:  # noqa: BLE001 - show a friendly message instead of crashing
            return None, "The file could not be read as a CSV. Please check its format."
    return None, "The file could not be read. Please save it as a UTF-8 CSV and try again."


def page_upload():
    page_header("Upload Data", "Bring your own CSV file and analyse it instantly.")
    show_flash()
    data_banner()

    st.markdown("#### Step 1 - Get the template")
    st.write(
        "Your file needs these columns: `date`, `sleep_hours`, `exercise_minutes`, `water_litres`, "
        "`screen_hours`, `study_hours`, `energy`. The columns `mood` and `sleep_quality` are optional."
    )
    st.download_button("⬇️ Download sample CSV template", data=du.template_csv_bytes(),
                       file_name="lifestat_template.csv", mime="text/csv")

    st.markdown("#### Step 2 - Upload your file")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("🌱 No file selected yet.")
        return

    raw, read_error = read_uploaded_csv(uploaded)
    if read_error:
        st.error(read_error)
        return

    result = du.validate_dataframe(raw)
    for message in result.errors:
        st.error(message)
    if not result.ok:
        return
    for message in result.warnings:
        st.warning(message)

    st.success(f"✅ {len(result.data)} valid records found.")
    st.dataframe(du.display_frame(result.data.head(15)), hide_index=True)
    if len(result.data) > 15:
        st.caption(f"Showing the first 15 of {len(result.data)} records.")

    st.markdown("#### Step 3 - Use the data")
    mode = st.radio(
        "What should LifeStat do with these records?",
        ["Replace my current data", "Add to my current data"],
        horizontal=True,
    )
    if st.button("✅ Use this data"):
        replaced = store_records(result.data, replace_all=(mode == "Replace my current data"))
        message = f"{len(result.data)} records loaded."
        if replaced:
            message += f" {replaced} existing date(s) were updated."
        set_flash("success", message)
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Sleep & Energy Insights  (joint probability)
# ---------------------------------------------------------------------------
def event_picker(prefix, title, default_col, default_op, container):
    with container:
        st.markdown(f"**{title}**")
        col = st.selectbox("Lifestyle factor", du.NUMERIC_COLUMNS,
                           index=du.NUMERIC_COLUMNS.index(default_col),
                           format_func=lambda c: du.LABELS[c], key=f"{prefix}_col")
        ops = list(CONDITIONS)
        op = st.selectbox("Condition", ops, index=ops.index(default_op),
                          format_func=lambda o: CONDITIONS[o], key=f"{prefix}_op")
        low, high = du.VALID_RANGES[col]
        value = st.number_input("Value", min_value=float(low), max_value=float(high),
                                value=float(DEFAULT_THRESHOLD[col]), step=float(STEP[col]),
                                key=f"{prefix}_val_{col}")
    return col, op, float(value)


def event_text(col, op, value):
    return f"{du.LABELS[col]} is {CONDITIONS[op]} {value:g}"


def venn_figure(res):
    fig = go.Figure()
    fig.add_shape(type="circle", x0=-2.0, y0=-1.3, x1=0.6, y1=1.3, fillcolor=GREEN,
                  opacity=0.45, line=dict(color=GREEN_DARK, width=2))
    fig.add_shape(type="circle", x0=-0.6, y0=-1.3, x1=2.0, y1=1.3, fillcolor=GOLD,
                  opacity=0.45, line=dict(color="#B98A1F", width=2))
    for x, y, text, size in [
        (-1.35, 0.0, f"<b>{res['a_only']}</b><br>A only", 15),
        (0.0, 0.0, f"<b>{res['count_both']}</b><br>Both", 17),
        (1.35, 0.0, f"<b>{res['b_only']}</b><br>B only", 15),
        (0.0, -1.75, f"Neither: {res['neither']} days", 13),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=size, color=TEXT))
    fig.add_annotation(x=-1.35, y=1.6, text="<b>Event A</b>", showarrow=False, font=dict(color=GREEN_DARK))
    fig.add_annotation(x=1.35, y=1.6, text="<b>Event B</b>", showarrow=False, font=dict(color="#8A6510"))
    fig.update_xaxes(visible=False, range=[-2.4, 2.4])
    fig.update_yaxes(visible=False, range=[-2.1, 1.9], scaleanchor="x", scaleratio=1)
    return style_fig(fig, "Days with Event A, Event B, or both", height=360)


def page_sleep_energy():
    page_header("Sleep & Energy Insights",
                "How often do two good habits or outcomes happen on the same day?")
    show_flash()
    data_banner()
    if not require_records(1):
        return
    df = st.session_state.df

    c1, c2 = st.columns(2)
    col_a, op_a, val_a = event_picker("ev_a", "Event A", "sleep_hours", ">=", c1)
    col_b, op_b, val_b = event_picker("ev_b", "Event B", "energy", ">=", c2)

    res = se.joint_probability(df, col_a, op_a, val_a, col_b, op_b, val_b)
    if res is None:
        empty_state("There is no usable data for these two factors.")
        return

    st.markdown(f"**Event A:** {event_text(col_a, op_a, val_a)}  \n**Event B:** {event_text(col_b, op_b, val_b)}")

    cards = st.columns(5)
    with cards[0]:
        metric_card("Total days", f"{res['total']}")
    with cards[1]:
        metric_card("Event A days", f"{res['count_a']}")
    with cards[2]:
        metric_card("Event B days", f"{res['count_b']}")
    with cards[3]:
        metric_card("Both together", f"{res['count_both']}")
    with cards[4]:
        metric_card("Chance of both", f"{res['probability']:.1%}", "P(A ∩ B)")

    show_fig(venn_figure(res))

    st.markdown(
        f"**How it is calculated:** P(A ∩ B) = days with both ÷ total days = "
        f"{res['count_both']} ÷ {res['total']} = **{res['probability']:.3f}**"
    )
    if res["count_both"] == 0:
        text = "In your records, these two things <b>never happened on the same day</b>."
    else:
        text = (f"On <b>{res['count_both']} of {res['total']} recorded days ({res['probability']:.1%})</b>, "
                "both events happened together.")
    insight(text)
    if res["total"] < 20:
        st.caption("⚠️ With fewer than 20 records this estimate can change a lot as you add days.")


# ---------------------------------------------------------------------------
# PAGE: Energy Patterns  (discrete distribution)
# ---------------------------------------------------------------------------
def page_energy_patterns():
    page_header("Energy Patterns", "How are your daily energy ratings spread out, and what is typical?")
    show_flash()
    data_banner()
    if not require_records(se.MIN_RECORDS):
        return
    res = se.discrete_distribution(st.session_state.df["energy"])
    table = res["table"]

    cards = st.columns(4)
    with cards[0]:
        metric_card("Expected energy", fmt(res["expected"]), "E(X), the long-run average")
    with cards[1]:
        metric_card("Variance", fmt(res["variance"]), "Var(X)")
    with cards[2]:
        metric_card("Standard deviation", fmt(res["std"]), "typical spread")
    with cards[3]:
        metric_card("Days analysed", f"{res['n']}")

    fig = go.Figure(go.Bar(x=table["Rating"], y=table["Probability"], marker_color=GREEN,
                           text=[f"{p:.0%}" for p in table["Probability"]], textposition="outside",
                           name="Probability"))
    fig.add_vline(x=res["expected"], line_dash="dash", line_color=GOLD,
                  annotation_text=f"Expected = {res['expected']:.2f}")
    fig.update_xaxes(title="Energy rating", dtick=1)
    fig.update_yaxes(title="Probability", rangemode="tozero")
    show_fig(style_fig(fig, "Chance of each energy rating"))

    with st.expander("See the full table and the working", expanded=False):
        shown = table.copy()
        shown["Probability"] = shown["Probability"].round(4)
        shown["x * P(x)"] = shown["x * P(x)"].round(4)
        st.dataframe(shown, hide_index=True)
        st.markdown(
            f"- **Expected value:** E(X) = Σ x·P(X = x) = **{res['expected']:.3f}**\n"
            f"- **Average of squares:** E(X²) = Σ x²·P(X = x) = {res['expected_square']:.3f}\n"
            f"- **Variance:** Var(X) = E(X²) − [E(X)]² = {res['expected_square']:.3f} − "
            f"{res['expected'] ** 2:.3f} = **{res['variance']:.3f}**\n"
            f"- **Standard deviation:** √Var(X) = **{res['std']:.3f}**"
        )

    common = ", ".join(str(v) for v in res["most_common"])
    insight(
        f"Your most common energy rating is <b>{common}</b>. Over the long run your energy averages about "
        f"<b>{res['expected']:.1f} out of 10</b>, and a day usually lands within about "
        f"<b>{res['std']:.1f} points</b> of that. You rated <b>{res['p_high']:.0%}</b> of days as 7 or higher."
    )


# ---------------------------------------------------------------------------
# PAGE: Sleep Duration Insights  (continuous distribution)
# ---------------------------------------------------------------------------
def page_sleep_duration():
    page_header("Sleep Duration Insights",
                "See how a measurement such as sleep is spread across your days.")
    show_flash()
    data_banner()
    if not require_records(se.MIN_RECORDS):
        return

    options = ["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours"]
    col = st.selectbox("Choose a measurement", options, format_func=lambda c: du.LABELS[c])
    summary = se.continuous_summary(st.session_state.df[col])
    x = summary["values"]
    check = se.normal_fit_check(x)

    cards = st.columns(4)
    with cards[0]:
        metric_card("Mean", fmt(summary["mean"]), "average")
    with cards[1]:
        metric_card("Variance", fmt(summary["variance"]), "sample variance (n − 1)")
    with cards[2]:
        metric_card("Standard deviation", fmt(summary["std"]), "typical spread")
    with cards[3]:
        metric_card("Days analysed", f"{summary['n']}")

    # Range chosen by the user
    st.markdown("#### Chance of falling in a range")
    default_low, default_high = DEFAULT_RANGE.get(
        col, (max(0.0, round(summary["mean"] - summary["std"], 1)), round(summary["mean"] + summary["std"], 1))
    )
    r1, r2 = st.columns(2)
    with r1:
        low = st.number_input("From (lower value)", min_value=0.0, value=float(default_low),
                              step=0.1, key=f"range_low_{col}")
    with r2:
        high = st.number_input("To (upper value)", min_value=0.0, value=float(default_high),
                               step=0.1, key=f"range_high_{col}")
    if low > high:
        st.warning("The lower value was larger than the upper value, so they were swapped.")
        low, high = high, low

    use_normal = st.checkbox(
        "Also show an estimate using a fitted bell curve (normal distribution)",
        value=check["reasonable"],
    )

    empirical = se.empirical_probability_between(x, low, high)
    normal_p = se.normal_probability_between(summary["mean"], summary["std"], low, high) if use_normal else None

    # Density chart
    curve = se.density_curve(x)
    bins = int(min(20, max(5, np.sqrt(len(x)))))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=bins, name="Your data",
                               marker_color=GREEN_LIGHT, opacity=0.75))
    if curve is not None:
        grid, density = curve
        fig.add_trace(go.Scatter(x=grid, y=density, mode="lines", name="Smoothed density",
                                 line=dict(color=GREEN_DARK, width=3)))
        inside = (grid >= low) & (grid <= high)
        if inside.any():
            fig.add_trace(go.Scatter(x=grid[inside], y=density[inside], mode="lines", fill="tozeroy",
                                     name="Selected range", line=dict(color=GOLD, width=1),
                                     fillcolor="rgba(233,180,76,0.45)"))
    if use_normal and summary["std"] > 0:
        grid_n = np.linspace(summary["min"] - summary["std"], summary["max"] + summary["std"], 200)
        pdf = np.exp(-0.5 * ((grid_n - summary["mean"]) / summary["std"]) ** 2) / (
            summary["std"] * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=grid_n, y=pdf, mode="lines", name="Fitted bell curve",
                                 line=dict(color=CLAY, width=2, dash="dash")))
    fig.update_xaxes(title=du.LABELS[col])
    fig.update_yaxes(title="Density")
    show_fig(style_fig(fig, "How your values are spread (probability density)", height=380))
    if curve is None:
        st.info("All recorded values are almost identical, so a smooth density curve cannot be drawn.")

    # Cumulative chart
    xs, ys = se.empirical_cdf(x)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Your data",
                              line=dict(color=GREEN_DARK, width=3, shape="hv")))
    if use_normal and summary["std"] > 0:
        from scipy import stats as sp_stats

        grid_n = np.linspace(summary["min"] - summary["std"], summary["max"] + summary["std"], 200)
        fig2.add_trace(go.Scatter(x=grid_n, y=sp_stats.norm.cdf(grid_n, summary["mean"], summary["std"]),
                                  mode="lines", name="Fitted bell curve",
                                  line=dict(color=CLAY, width=2, dash="dash")))
    fig2.add_vrect(x0=low, x1=high, fillcolor=GOLD, opacity=0.2, line_width=0)
    fig2.update_xaxes(title=du.LABELS[col])
    fig2.update_yaxes(title="Share of days at or below", range=[0, 1.02], tickformat=".0%")
    show_fig(style_fig(fig2, "Running total of days (cumulative distribution)", height=340))

    # Results
    res_cols = st.columns(2)
    with res_cols[0]:
        metric_card("From your records", f"{empirical['probability']:.1%}",
                    f"{empirical['count']} of {empirical['total']} days between {low:g} and {high:g}")
    with res_cols[1]:
        if use_normal:
            if normal_p is None:
                metric_card("Fitted bell-curve estimate", "—", "needs some variation in the data")
            else:
                metric_card("Fitted bell-curve estimate", f"{normal_p:.1%}",
                            "assumes a normal distribution")
        else:
            metric_card("Fitted bell-curve estimate", "Off", "tick the box above to show it")

    st.markdown("#### Assumptions")
    p_text = "n/a" if check["p_value"] is None else f"{check['p_value']:.3f}"
    st.markdown(
        "- **From your records** counts real days in the range. It makes no assumptions about the data's shape.\n"
        "- **Fitted bell curve** assumes the data is symmetric and bell-shaped, using your mean and standard "
        "deviation. It can give probability to impossible values (for example, negative hours).\n"
        f"- **Bell-curve check (Shapiro-Wilk test):** p-value = {p_text}. {check['reason']}\n"
        "- Not every lifestyle measurement is bell-shaped, so LifeStat never assumes it automatically."
    )
    if empirical["total"] < 20:
        st.caption("⚠️ With fewer than 20 records, these estimates are rough.")


# ---------------------------------------------------------------------------
# PAGE: Lifestyle Relationships  (regression lines)
# ---------------------------------------------------------------------------
def page_relationships():
    page_header("Lifestyle Relationships",
                "Explore how one habit is linked to another and try simple what-if estimates.")
    show_flash()
    data_banner()
    if not require_records(se.MIN_RECORDS):
        return
    df = st.session_state.df

    c1, c2 = st.columns(2)
    with c1:
        x_col = numeric_choice("Variable X (horizontal)", "reg_x", "sleep_hours")
    with c2:
        y_col = numeric_choice("Variable Y (vertical)", "reg_y", "energy")
    if x_col == y_col:
        st.warning("Please choose two different variables.")
        return

    x_name, y_name = du.SHORT_NAMES[x_col], du.SHORT_NAMES[y_col]
    res = se.regression_analysis(df[x_col], df[y_col], x_name, y_name)
    if not res["ok"]:
        st.info("🌱 " + res["message"])
        return

    cards = st.columns(4)
    with cards[0]:
        metric_card(f"{y_name} on {x_name}", fmt(res["b_yx"], 3), "coefficient b_yx")
    with cards[1]:
        metric_card(f"{x_name} on {y_name}", fmt(res["b_xy"], 3), "coefficient b_xy")
    with cards[2]:
        metric_card("R-squared", fmt(res["r_squared"], 3), "fit of the line")
    with cards[3]:
        metric_card("Days analysed", f"{res['n']}")

    # Scatter plot with both regression lines
    xs, ys = res["x"], res["y"]
    x_line = np.linspace(xs.min(), xs.max(), 50)
    y_line = res["a_yx"] + res["b_yx"] * x_line
    y_grid = np.linspace(ys.min(), ys.max(), 50)
    x_from_y = res["a_xy"] + res["b_xy"] * y_grid

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Your days",
                             marker=dict(color=GREEN, size=8, opacity=0.7)))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"{y_name} on {x_name}",
                             line=dict(color=GREEN_DARK, width=3)))
    fig.add_trace(go.Scatter(x=x_from_y, y=y_grid, mode="lines", name=f"{x_name} on {y_name}",
                             line=dict(color=CLAY, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=[res["x_mean"]], y=[res["y_mean"]], mode="markers", name="Average point",
                             marker=dict(color=GOLD, size=14, symbol="diamond",
                                         line=dict(color=TEXT, width=1))))
    fig.update_xaxes(title=du.LABELS[x_col])
    fig.update_yaxes(title=du.LABELS[y_col])
    show_fig(style_fig(fig, "Scatter plot with both regression lines", height=420))

    st.markdown("#### Regression equations")
    st.latex(r"Y - \bar{Y} = b_{yx}\,(X - \bar{X}) \qquad X - \bar{X} = b_{xy}\,(Y - \bar{Y})")
    sign_yx = "+" if res["b_yx"] >= 0 else "−"
    sign_xy = "+" if res["b_xy"] >= 0 else "−"
    st.markdown(
        f"- **{y_name} on {x_name}:** {y_name} = {res['a_yx']:.3f} {sign_yx} {abs(res['b_yx']):.3f} × {x_name}  \n"
        f"  (b_yx = {res['b_yx']:.4f}; the line passes through the averages "
        f"X̄ = {res['x_mean']:.2f}, Ȳ = {res['y_mean']:.2f})\n"
        f"- **{x_name} on {y_name}:** {x_name} = {res['a_xy']:.3f} {sign_xy} {abs(res['b_xy']):.3f} × {y_name}  \n"
        f"  (b_xy = {res['b_xy']:.4f})\n"
        f"- **R² = {res['r_squared']:.3f}**, which equals b_yx × b_xy = {res['b_yx'] * res['b_xy']:.3f}."
    )

    # What-if simulator
    st.markdown("#### What-if estimator")
    direction = st.radio(
        "What would you like to estimate?",
        [f"Estimate {y_name} from {x_name}", f"Estimate {x_name} from {y_name}"],
        horizontal=True,
    )
    from_y = direction.startswith(f"Estimate {x_name}")
    given_col, given_name = (y_col, y_name) if from_y else (x_col, x_name)
    target_col, target_name = (x_col, x_name) if from_y else (y_col, y_name)
    given_mean = res["y_mean"] if from_y else res["x_mean"]
    given_values = ys if from_y else xs
    low, high = du.VALID_RANGES[given_col]
    given = st.number_input(f"If {given_name} is...", min_value=float(low), max_value=float(high),
                            value=float(round(min(max(given_mean, low), high), 2)),
                            step=float(STEP[given_col]), key=f"whatif_{x_col}_{y_col}_{int(from_y)}")
    estimate = se.predict_x_from_y(res, given) if from_y else se.predict_y_from_x(res, given)
    unit_note = ""
    t_low, t_high = du.VALID_RANGES[target_col]
    if estimate < t_low or estimate > t_high:
        unit_note = (f" This estimate is outside the possible range for {target_name.lower()} "
                     f"({t_low:g} to {t_high:g}), so treat it with caution.")
    insight(f"If {given_name.lower()} is <b>{given:g}</b>, the estimated {target_name.lower()} is "
            f"<b>{estimate:.2f}</b>.{unit_note}")
    if given < given_values.min() or given > given_values.max():
        st.caption("⚠️ That value is outside the range in your records, so the estimate is less reliable.")

    note("ℹ️ <b>Remember:</b> a regression line shows an <i>estimated relationship</i> in your records. "
         "It does <b>not</b> prove that one habit causes the other. Each line is built to predict a "
         "different variable, which is why the two lines are not the same.")


# ---------------------------------------------------------------------------
# PAGE: Habit Connections  (Pearson correlation)
# ---------------------------------------------------------------------------
def plain_language_correlation(x_name, y_name, res):
    r = res["r"]
    if res["direction"] == "no clear direction":
        return (f"There is <b>almost no straight-line link</b> between {x_name.lower()} and {y_name.lower()} "
                "in your records.")
    if r > 0:
        return (f"Days with more {x_name.lower()} tend to have <b>higher {y_name.lower()}</b>. "
                f"The link is <b>{res['strength']}</b>.")
    return (f"Days with more {x_name.lower()} tend to have <b>lower {y_name.lower()}</b>. "
            f"The link is <b>{res['strength']}</b>.")


def page_connections():
    page_header("Habit Connections",
                "Find out how strongly two habits move together.")
    show_flash()
    data_banner()
    if not require_records(se.MIN_RECORDS):
        return
    df = st.session_state.df

    c1, c2 = st.columns(2)
    with c1:
        x_col = numeric_choice("First habit", "cor_x", "screen_hours")
    with c2:
        y_col = numeric_choice("Second habit", "cor_y", "energy")
    if x_col == y_col:
        st.warning("Please choose two different habits.")
        return

    x_name, y_name = du.SHORT_NAMES[x_col], du.SHORT_NAMES[y_col]
    res = se.pearson_correlation(df[x_col], df[y_col], x_name, y_name)
    if not res["ok"]:
        st.info("🌱 " + res["message"])
        return

    cards = st.columns(4)
    with cards[0]:
        metric_card("Pearson correlation (r)", fmt(res["r"], 3), "from −1 to +1")
    with cards[1]:
        metric_card("Direction", res["direction"].capitalize(), "")
    with cards[2]:
        metric_card("Strength", res["strength"].capitalize(), "of the straight-line link")
    with cards[3]:
        metric_card("Days analysed", f"{res['n']}")

    x_line = np.linspace(res["x"].min(), res["x"].max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["x"], y=res["y"], mode="markers", name="Your days",
                             marker=dict(color=GREEN, size=8, opacity=0.7)))
    fig.add_trace(go.Scatter(x=x_line, y=res["a_yx"] + res["b_yx"] * x_line, mode="lines",
                             name="Trend line", line=dict(color=GOLD, width=3)))
    fig.update_xaxes(title=du.LABELS[x_col])
    fig.update_yaxes(title=du.LABELS[y_col])
    show_fig(style_fig(fig, f"{x_name} and {y_name}", height=400))

    insight(plain_language_correlation(x_name, y_name, res))
    st.latex(r"r = \frac{\sum (x-\bar{x})(y-\bar{y})}{\sqrt{\sum (x-\bar{x})^2 \, \sum (y-\bar{y})^2}}")
    st.markdown(f"Calculated value: **r = {res['r']:.4f}**")
    note("ℹ️ <b>Correlation does not imply causation.</b> Two habits can move together because of a "
         "third factor, or by chance. Pearson's r only measures <i>straight-line</i> links.")

    with st.expander("See all habit pairs at once"):
        corr = df[du.NUMERIC_COLUMNS].corr(method="pearson")
        names = [du.SHORT_NAMES[c] for c in du.NUMERIC_COLUMNS]
        z = corr.to_numpy()
        fig_h = go.Figure(go.Heatmap(
            z=z, x=names, y=names, zmin=-1, zmax=1,
            colorscale=[[0, CLAY], [0.5, CREAM], [1, GREEN_DARK]],
            text=np.round(z, 2), texttemplate="%{text}",
        ))
        fig_h.update_yaxes(autorange="reversed")
        show_fig(style_fig(fig_h, "Pearson correlation between all habits", height=420))


# ---------------------------------------------------------------------------
# PAGE: My Records
# ---------------------------------------------------------------------------
def page_records():
    page_header("My Records", "View, download, delete or reset your data.")
    show_flash()
    data_banner()
    df = st.session_state.df

    if len(df) == 0:
        empty_state()
    else:
        st.markdown(f"**{len(df)} records**")
        st.dataframe(du.display_frame(df), hide_index=True)

    st.download_button("⬇️ Download records as CSV", data=du.to_csv_bytes(df),
                       file_name="lifestat_records.csv", mime="text/csv", disabled=len(df) == 0)

    if len(df) > 0:
        st.markdown("#### Delete records")
        date_labels = df["date"].dt.strftime("%Y-%m-%d").tolist()
        chosen = st.multiselect("Select dates to delete", date_labels)
        if st.button("🗑️ Delete selected records", disabled=len(chosen) == 0):
            keep = ~pd.Series(date_labels).isin(chosen).to_numpy()
            st.session_state.df = df[keep].reset_index(drop=True)
            set_flash("success", f"{len(chosen)} record(s) deleted.")
            st.rerun()

    st.markdown("#### Reset")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reset to sample data"):
            st.session_state.df = du.load_sample()
            st.session_state.source = "sample"
            set_flash("info", "Sample data restored.")
            st.rerun()
    with c2:
        confirm = st.checkbox("I want to remove all records")
        if st.button("🧹 Clear all records", disabled=not confirm):
            st.session_state.df = du.empty_dataframe()
            st.session_state.source = "user"
            set_flash("info", "All records cleared. You can start fresh with Daily Entry or Upload Data.")
            st.rerun()
    note("💡 Records are kept only for your current session. Download a CSV to keep a permanent copy.")


# ---------------------------------------------------------------------------
# PAGE: Reports
# ---------------------------------------------------------------------------
def page_reports():
    page_header("Reports", "Create a readable summary of your lifestyle statistics.")
    show_flash()
    data_banner()
    df = st.session_state.df
    if len(df) == 0:
        empty_state()
        return

    sections = st.multiselect("Choose what to include", rb.ALL_SECTIONS, default=rb.ALL_SECTIONS)
    if not sections:
        st.info("Select at least one section to build a report.")
        return

    report_text = rb.build_report(df, sections, st.session_state.source)
    st.text_area("Report preview", value=report_text, height=420, disabled=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download report (TXT)", data=report_text.encode("utf-8"),
                           file_name="lifestat_report.txt", mime="text/plain")
    with c2:
        summary_csv = rb.summary_csv_table(df).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download key results (CSV)", data=summary_csv,
                           file_name="lifestat_summary.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# PAGE: About LifeStat
# ---------------------------------------------------------------------------
def page_about():
    page_header("About LifeStat", "Personal lifestyle analysis powered by probability and statistics.")
    show_flash()

    st.markdown("#### Purpose")
    st.write(
        "LifeStat helps you understand everyday habits such as sleep, exercise, water intake, screen time "
        "and energy. You can enter records by hand or upload a CSV file, and LifeStat turns them into "
        "charts and statistical insights. It is a 3rd-semester engineering Mathematics project based on "
        "Probability and Statistics."
    )

    st.markdown("#### Existing systems vs LifeStat")
    st.write(
        "Many lifestyle tracking apps concentrate on recording and monitoring. LifeStat is built around "
        "analysing the relationships and patterns in that data. The table compares the typical focus of "
        "each approach; it does not claim these features are missing from every existing app."
    )
    comparison = pd.DataFrame(
        {
            "Area": ["Main purpose", "Summaries", "Patterns in data", "Chances of events", "Spread of values",
                     "Relationships between habits", "Charts"],
            "Conventional lifestyle trackers (typical focus)": [
                "Basic activity tracking and habit monitoring",
                "Daily, weekly and monthly averages and general summaries",
                "Progress over time",
                "Usually not the focus",
                "Usually shown as averages",
                "Usually not the focus",
                "Basic progress charts",
            ],
            "LifeStat (proposed system)": [
                "Statistical understanding of lifestyle data",
                "Averages plus variance and standard deviation",
                "Relationship-based analysis",
                "Probability-based insights on events happening together",
                "Distribution analysis of values, not just averages",
                "Regression lines with what-if estimates, and Pearson correlation",
                "Interactive statistical visualisations",
            ],
        }
    ).set_index("Area")
    st.table(comparison)
    st.write(
        "**The unique combination:** LifeStat joins personal lifestyle tracking with five mathematical and "
        "statistical methods in a single interactive application, using an academic approach."
    )

    st.markdown("#### The five analyses behind the scenes")
    with st.expander("Sleep & Energy Insights (Joint probability)"):
        st.latex(r"P(A \cap B) = \frac{\text{observations satisfying both events}}{\text{total observations}}")
        st.write("Counts how often two chosen conditions happen on the same day.")
    with st.expander("Energy Patterns (Discrete probability distribution)"):
        st.latex(r"E(X)=\sum x\,P(X=x) \qquad Var(X)=E(X^2)-[E(X)]^2")
        st.write("Treats the daily energy rating (1-10) as a discrete variable and finds its probability of each value.")
    with st.expander("Sleep Duration Insights (Continuous probability distribution)"):
        st.write(
            "Uses the data's own empirical distribution and a smoothed density estimate. A fitted normal curve "
            "is offered only as an option, with its assumptions shown. Variance uses the sample formula (n − 1)."
        )
    with st.expander("Lifestyle Relationships (Regression lines)"):
        st.latex(r"Y-\bar{Y}=b_{yx}(X-\bar{X}) \qquad X-\bar{X}=b_{xy}(Y-\bar{Y})")
        st.latex(r"b_{yx}=\frac{\sum (x-\bar{x})(y-\bar{y})}{\sum (x-\bar{x})^2} \qquad "
                 r"b_{xy}=\frac{\sum (x-\bar{x})(y-\bar{y})}{\sum (y-\bar{y})^2}")
        st.write("Gives two lines and R², and lets you make simple what-if estimates.")
    with st.expander("Habit Connections (Pearson correlation coefficient)"):
        st.latex(r"r = \frac{\sum (x-\bar{x})(y-\bar{y})}{\sqrt{\sum (x-\bar{x})^2 \sum (y-\bar{y})^2}}")
        st.write("Measures the strength and direction of a straight-line link between two habits.")

    st.markdown("#### Privacy and data storage")
    st.write(
        "LifeStat asks only for lifestyle measurements, with no names or contact details. Records are held in "
        "your browser session. They are not saved permanently, so download a CSV copy to keep your data."
    )
    st.markdown("#### Limitations")
    st.write(
        "- Results describe only the records you provide, and small datasets give rough estimates.\n"
        "- Regression and correlation show relationships, not causes.\n"
        "- LifeStat is an educational project and does not give medical advice."
    )
    st.markdown("#### Built with")
    st.write("Python, Streamlit, Pandas, NumPy, SciPy and Plotly. No paid services, AI or machine learning.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Overview": page_overview,
    "✍️ Daily Entry": page_daily_entry,
    "📤 Upload Data": page_upload,
    "🌙 Sleep & Energy Insights": page_sleep_energy,
    "⚡ Energy Patterns": page_energy_patterns,
    "😴 Sleep Duration Insights": page_sleep_duration,
    "🔗 Lifestyle Relationships": page_relationships,
    "🌿 Habit Connections": page_connections,
    "🗂️ My Records": page_records,
    "📄 Reports": page_reports,
    "ℹ️ About LifeStat": page_about,
}


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    with st.sidebar:
        st.markdown('<div class="brand">🌿 LifeStat</div>'
                    '<div class="brand-sub">Personal Lifestyle Analyzer</div>', unsafe_allow_html=True)
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        source_label = {"sample": "Sample data", "mixed": "Sample + your records", "user": "Your records"}
        st.caption(f"📊 {len(st.session_state.df)} days · {source_label[st.session_state.source]}")

    PAGES[choice]()


main()
