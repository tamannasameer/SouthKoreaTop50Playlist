import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlantic | South Korea Top 50 Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS — clean, minimal, feedback-addressed
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

    /* KPI card — large readable number, no colour fill */
    .kpi-wrap {
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px 14px;
        background: #0F172A;
        text-align: center;
    }
    .kpi-label { font-size:.72rem; color:#94A3B8; text-transform:uppercase;
                 letter-spacing:.07em; margin-bottom:4px; }
    .kpi-value { font-size:2.1rem; font-weight:800; color:#F1F5F9; line-height:1.1; }
    .kpi-sub   { font-size:.75rem; color:#64748B; margin-top:4px; }

    /* Section header */
    .sec-head {
        font-size:.75rem; font-weight:700; color:#818CF8;
        text-transform:uppercase; letter-spacing:.08em;
        border-bottom:1px solid #1E293B; padding-bottom:6px;
        margin: 1.4rem 0 .8rem;
    }

    /* Divider */
    .hdiv { border:none; border-top:1px solid #1E293B; margin:1rem 0; }

    /* Data note */
    .data-note {
        background:#0F172A; border-left:3px solid #818CF8;
        padding:9px 14px; border-radius:0 8px 8px 0;
        font-size:.8rem; color:#94A3B8; margin-bottom:1rem;
    }

    /* Legend label — overrides plotly raw var names */
    .legend-note { font-size:.75rem; color:#64748B; margin-top:-8px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
C_PURPLE  = "#818CF8"
C_PINK    = "#F472B6"
C_TEAL    = "#2DD4BF"
C_AMBER   = "#FBBF24"
C_GREEN   = "#34D399"
C_RED     = "#F87171"
C_BLUE    = "#60A5FA"
C_SLATE   = "#94A3B8"

ALBUM_COLORS   = {"single": C_PURPLE, "album": C_TEAL, "compilation": C_AMBER}
EXPLICIT_COLORS= {True: C_RED, False: C_GREEN}
STINT_COLORS   = [C_BLUE, C_PURPLE, C_PINK, C_AMBER, C_GREEN, C_TEAL, C_RED, C_SLATE]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    margin=dict(l=10, r=10, t=40, b=10),
)

def styled_fig(fig, height=380, legend=True):
    fig.update_layout(
        **PLOTLY_BASE,
        height=height,
        legend=dict(
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="#334155",
            borderwidth=1,
            font_size=11,
        ) if legend else dict(visible=False),
    )
    fig.update_xaxes(gridcolor="#1E293B", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1E293B", zerolinecolor="#334155")
    return fig

# ─────────────────────────────────────────────────────────────
#  DATA LOADING  (Phases 1 + 2 already applied)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data…")
def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    # Phase 1 clean
    df = df.drop_duplicates()
    df = (df.sort_values(['date','position'])
            .drop_duplicates(subset=['date','song','artist'], keep='first')
            .reset_index(drop=True))

    # Phase 2 base features
    df['duration_min'] = (df['duration_ms'] / 60000).round(2)
    df['song_key']     = df['song'].str.strip() + ' || ' + df['artist'].str.strip()
    df = df.sort_values(['song_key','date']).reset_index(drop=True)

    # Re-entry detection
    records = []
    for key, group in df.groupby('song_key', sort=False):
        group = group.sort_values('date').reset_index(drop=True)
        dates = group['date'].tolist()
        stint, stints = 0, [0]
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days > 1:
                stint += 1
            stints.append(stint)
        group['stint'] = stints
        for s in group['stint'].unique():
            sr = group[group['stint'] == s]
            group.loc[group['stint'] == s, 'days_on_chart_stint'] = len(sr)
            if s == 0:
                group.loc[group['stint'] == s, 'gap_days'] = 0
            else:
                prev_last  = group[group['stint'] == s-1]['date'].max()
                curr_first = sr['date'].min()
                group.loc[group['stint'] == s, 'gap_days'] = max((curr_first - prev_last).days - 1, 0)
        group['re_entry_number'] = group['stint']
        group['is_reentry_day']  = False
        for s in group['stint'].unique():
            if s > 0:
                fd = group[group['stint'] == s]['date'].min()
                group.loc[(group['stint'] == s) & (group['date'] == fd), 'is_reentry_day'] = True
        records.append(group)

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(['song_key','date']).reset_index(drop=True)

    # Rank / popularity change
    df['prev_position']   = df.groupby('song_key')['position'].shift(1)
    df['rank_change']     = (df['prev_position'] - df['position']).fillna(0)
    df['prev_popularity'] = df.groupby('song_key')['popularity'].shift(1)
    df['pop_change']      = (df['popularity'] - df['prev_popularity'].fillna(df['popularity'])).round(2)

    # Momentum spike score
    df['momentum_spike_score'] = 0.0
    mask = df['is_reentry_day']
    df.loc[mask, 'momentum_spike_score'] = (
        df.loc[mask, 'pop_change'].clip(lower=0) * 0.5 +
        (51 - df.loc[mask, 'position']) * 0.5
    ).round(2)

    # Fandom intensity score per song
    ss = df.groupby('song_key').agg(
        total_reentries   = ('re_entry_number', 'max'),
        max_momentum      = ('momentum_spike_score', 'max'),
        avg_popularity    = ('popularity', 'mean'),
    ).reset_index()
    def mm(s): return ((s - s.min()) / (s.max() - s.min()) * 100).fillna(0).round(1)
    ss['fandom_intensity_score'] = (mm(ss['total_reentries'])*0.40 +
                                    mm(ss['max_momentum'])*0.35 +
                                    mm(ss['avg_popularity'])*0.25).round(1)
    df = df.merge(ss[['song_key','fandom_intensity_score']], on='song_key', how='left')
    return df


@st.cache_data(show_spinner=False)
def build_song_summary(df):
    """One-row-per-song aggregation used across all modules."""
    agg = df.groupby(['song_key','song','artist','album_type','is_explicit']).agg(
        total_appearances       = ('date','count'),
        total_reentries         = ('re_entry_number','max'),
        total_stints            = ('stint','max'),
        avg_popularity          = ('popularity','mean'),
        peak_rank               = ('position','min'),
        avg_rank                = ('position','mean'),
        max_momentum_spike      = ('momentum_spike_score','max'),
        avg_days_per_stint      = ('days_on_chart_stint','mean'),
        avg_gap_days            = ('gap_days', lambda x: x[x>0].mean() if (x>0).any() else 0),
        fandom_intensity_score  = ('fandom_intensity_score','first'),
        duration_min            = ('duration_min','first'),
        total_tracks            = ('total_tracks','first'),
        first_entry             = ('date','min'),
        last_seen               = ('date','max'),
    ).reset_index()
    agg['avg_popularity'] = agg['avg_popularity'].round(1)
    agg['avg_rank']       = agg['avg_rank'].round(1)
    agg['avg_days_per_stint'] = agg['avg_days_per_stint'].round(1)
    agg['avg_gap_days']       = agg['avg_gap_days'].round(1)
    return agg


# ─────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────
DATA_PATH = "Atlantic_South_Korea.csv"
df_raw   = load_data(DATA_PATH)
song_sum = build_song_summary(df_raw)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Atlantic Records")
    st.markdown("**South Korea Top 50 — Comeback & Fandom Analysis**")
    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    st.markdown('<p class="sec-head">📅 Date Range</p>', unsafe_allow_html=True)
    min_d = df_raw['date'].min().date()
    max_d = df_raw['date'].max().date()
    _dr = st.date_input("Date range", value=(min_d, max_d),
                         min_value=min_d, max_value=max_d,
                         label_visibility="collapsed")
    if isinstance(_dr, (list, tuple)) and len(_dr) == 2:
        d_from, d_to = _dr
    else:
        st.stop()

    st.markdown('<p class="sec-head">🎤 Artist</p>', unsafe_allow_html=True)
    all_artists = sorted(df_raw['artist'].unique())
    sel_artists = st.multiselect("Artists", all_artists, default=all_artists,
                                  label_visibility="collapsed")

    st.markdown('<p class="sec-head">💿 Album Type</p>', unsafe_allow_html=True)
    sel_atype = st.multiselect("Album type",
                                sorted(df_raw['album_type'].unique()),
                                default=sorted(df_raw['album_type'].unique()),
                                label_visibility="collapsed")

    st.markdown('<p class="sec-head">🔄 Min Re-Entries</p>', unsafe_allow_html=True)
    max_re = int(df_raw['re_entry_number'].max())
    min_reentry = st.slider("Minimum re-entries", 0, max_re, 0,
                             label_visibility="collapsed")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
    st.caption("Atlantic Recording Corporation\nSouth Korea Market Intelligence")

# ─────────────────────────────────────────────────────────────
#  APPLY FILTERS
# ─────────────────────────────────────────────────────────────
df = df_raw.copy()
df = df[(df['date'].dt.date >= d_from) & (df['date'].dt.date <= d_to)]
if sel_artists:
    df = df[df['artist'].isin(sel_artists)]
if sel_atype:
    df = df[df['album_type'].isin(sel_atype)]

ss = build_song_summary(df)
ss = ss[ss['total_reentries'] >= min_reentry]
filtered_keys = ss['song_key'].tolist()
df = df[df['song_key'].isin(filtered_keys)]

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🎵 South Korea Top 50 — Comeback & Fandom Intelligence")
st.markdown("**Atlantic Recording Corporation** · Chart Re-Entry, Momentum Spike & Fandom Intensity Analysis")

st.markdown(f"""
<div class="data-note">
  📋 <strong>Dataset:</strong> {len(df_raw):,} playlist snapshots ·
  {df_raw['date'].nunique()} unique dates ({df_raw['date'].min().strftime('%b %Y')} – {df_raw['date'].max().strftime('%b %Y')}) ·
  {df_raw['song_key'].nunique()} unique songs · {df_raw['artist'].nunique()} artists.
  One duplicate date (2025-03-01 had 100 entries — cleaned to 50 by removing exact duplicates and keeping first occurrence).
  Dates are clean with no temporal anomaly.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TOP KPI ROW
# ─────────────────────────────────────────────────────────────
total_songs   = ss['song_key'].nunique()
total_reentry = ss['total_reentries'].sum()
reentry_songs = (ss['total_reentries'] > 0).sum()
avg_gap       = ss['avg_gap_days'].replace(0, np.nan).mean()
avg_retention = ss['avg_days_per_stint'].mean()
top_fandom    = ss.nlargest(1, 'fandom_intensity_score').iloc[0]['song'].split('(')[0].strip() if len(ss) else "—"

k1,k2,k3,k4,k5,k6 = st.columns(6)
avg_gap_val    = f"{avg_gap:.0f}" if not (pd.isna(avg_gap) or np.isinf(avg_gap)) else "—"
avg_ret_val    = f"{avg_retention:.1f}d" if not (pd.isna(avg_retention) or np.isinf(avg_retention)) else "—"

kpis = [
    (k1, "Unique Songs",         f"{total_songs:,}",       "in filtered view"),
    (k2, "Total Re-Entries",     f"{int(total_reentry):,}", "comeback events"),
    (k3, "Songs with Comebacks", f"{reentry_songs}",        f"{reentry_songs/max(total_songs,1)*100:.0f}% of songs"),
    (k4, "Avg Gap (days)",       avg_gap_val,               "off-chart before return"),
    (k5, "Avg Stint Length",     avg_ret_val,               "days per chart run"),
    (k6, "Top Fandom Song",      top_fandom[:16],           "highest intensity score"),
]
for col, label, value, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-wrap">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔄  Re-Entry Timeline",
    "⚡  Momentum Spikes",
    "🆚  Comeback vs First Entry",
    "🎸  Content Attributes",
    "🏆  Fandom Leaderboard",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — RE-ENTRY TIMELINE VISUALISER
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Re-Entry Timeline Visualiser")
    st.markdown("Each bar shows a song's chart run (stint). Gaps between bars = time off chart.")

    col_ctrl, col_info = st.columns([2,1], gap="large")
    with col_ctrl:
        top_n = st.slider("Show top N songs by re-entry count", 5, 30, 15,
                           label_visibility="visible")
        sort_by = st.radio("Sort by", ["Re-entry count", "First entry date", "Fandom intensity"],
                            horizontal=True)

    top_songs = ss.nlargest(top_n, 'total_reentries') if sort_by == "Re-entry count" \
                else ss.nsmallest(top_n, 'first_entry') if sort_by == "First entry date" \
                else ss.nlargest(top_n, 'fandom_intensity_score')

    top_keys = top_songs['song_key'].tolist()
    tl_df    = df[df['song_key'].isin(top_keys)].copy()

    if tl_df.empty:
        st.info("No chart data found. Try widening the date range or adjusting filters.")
    else:
        tl_df['song_label'] = tl_df['song_key'].apply(lambda x: x.split(' || ')[0][:28])
        tl_df['month']      = tl_df['date'].dt.to_period('M').dt.to_timestamp()

        # ── Panel 1: Summary bar chart ────────────────────────────
        st.markdown('<p class="sec-head">📊 Chart Presence Summary — First Entry vs Comeback Days</p>',
                    unsafe_allow_html=True)

        summary = (tl_df.groupby(['song_label','re_entry_number'])
                   .agg(days=('date','count'))
                   .reset_index())
        summary['type'] = summary['re_entry_number'].apply(
            lambda x: 'First Entry' if x == 0 else 'Comeback')
        bar_data = (summary.groupby(['song_label','type'])['days']
                    .sum().reset_index())

        # Comeback count label per song
        comeback_counts = (tl_df[tl_df['is_reentry_day']]
                           .groupby('song_label')['date'].count()
                           .reset_index(name='comeback_count'))

        # Keep order consistent
        ordered_labels = [k.split(' || ')[0][:28] for k in top_keys]
        bar_data = bar_data[bar_data['song_label'].isin(ordered_labels)]
        bar_data['song_label'] = pd.Categorical(bar_data['song_label'],
                                                 categories=ordered_labels[::-1],
                                                 ordered=True)
        bar_data = bar_data.sort_values('song_label')

        fig_bar = px.bar(
            bar_data, x='days', y='song_label',
            color='type',
            color_discrete_map={'First Entry': C_BLUE, 'Comeback': C_PINK},
            orientation='h',
            barmode='stack',
            labels={'days': 'Total Days on Chart', 'song_label': '',
                    'type': 'Chart Run'},
            text='days',
        )
        fig_bar.update_traces(textposition='inside', textfont_size=10,
                              textfont_color='white', insidetextanchor='middle')

        # Add comeback count annotations on right
        for label in ordered_labels:
            cc = comeback_counts[comeback_counts['song_label'] == label]['comeback_count']
            total = bar_data[bar_data['song_label'] == label]['days'].sum()
            count = int(cc.values[0]) if len(cc) > 0 else 0
            if count > 0:
                fig_bar.add_annotation(
                    x=total + 2, y=label,
                    text=f" {count} comebacks",
                    showarrow=False,
                    font=dict(size=10, color=C_PINK),
                    xanchor='left',
                )

        fig_bar.update_layout(
            **PLOTLY_BASE,
            height=max(300, len(ordered_labels) * 28 + 60),
            legend=dict(title='Chart Run', bgcolor='rgba(15,23,42,0.9)',
                        bordercolor='#334155', borderwidth=1,
                        orientation='h', y=1.06, x=0),
            xaxis=dict(title='Total Days on Chart', gridcolor='#1E293B', showgrid=True),
            yaxis=dict(tickfont_size=11),
            bargap=0.25,
        )
        st.plotly_chart(fig_bar, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("🔵 Blue = first chart run · 🩷 Pink = all comeback days combined · "
                   "Pink label = number of individual comeback events")

        st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

        # ── Panel 2: Monthly presence heatmap ─────────────────────
        st.markdown('<p class="sec-head">📅 Monthly Chart Presence Heatmap</p>',
                    unsafe_allow_html=True)

        heat = (tl_df.groupby(['song_label', 'month'])
                .agg(days=('date','count'),
                     comeback=('is_reentry_day','any'),
                     peak_rank=('position','min'))
                .reset_index())

        ordered_for_heat = [l for l in ordered_labels if l in heat['song_label'].values]
        months_sorted    = sorted(heat['month'].unique())
        month_labels     = [m.strftime("%b '%y") for m in months_sorted]

        # Build z matrix with encoded values:
        # 0 = off chart, 1 = regular, 2 = comeback month
        z_vals, z_text = [], []
        for label in ordered_for_heat:
            row_z, row_t = [], []
            for m in months_sorted:
                cell = heat[(heat['song_label']==label) & (heat['month']==m)]
                if cell.empty:
                    row_z.append(0)
                    row_t.append('')
                else:
                    days = int(cell['days'].values[0])
                    cb   = bool(cell['comeback'].values[0])
                    pk   = int(cell['peak_rank'].values[0])
                    row_z.append(2 if cb else 1)
                    row_t.append(f"↩#{pk}" if cb else f"#{pk}")
            z_vals.append(row_z)
            z_text.append(row_t)

        # Three-color scale: off chart | regular | comeback
        colorscale = [
            [0.00, '#0A1628'],   # 0 = off chart
            [0.33, '#0A1628'],
            [0.34, '#1D4ED8'],   # 1 = regular (blue)
            [0.66, '#1D4ED8'],
            [0.67, '#BE185D'],   # 2 = comeback (pink)
            [1.00, '#DB2777'],
        ]

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=month_labels,
            y=ordered_for_heat,
            text=z_text,
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=8, color='rgba(255,255,255,0.85)'),
            colorscale=colorscale,
            zmin=0, zmax=2,
            showscale=False,
            xgap=3, ygap=3,
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>",
        ))

        # Right-side comeback count
        re_lookup = {
            k.split(' || ')[0][:26]: int(row['total_reentries'])
            for k, row in ss.set_index('song_key').iterrows()
            if k.split(' || ')[0][:26] in ordered_for_heat
        }
        for label in ordered_for_heat:
            cnt = re_lookup.get(label, 0)
            fig_heat.add_annotation(
                x=1.01, y=label, xref='paper', yref='y',
                text=f"<b>{cnt}×</b>",
                font=dict(size=10, color='#EC4899'),
                showarrow=False, xanchor='left',
            )

        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor='#0A1628',
            font_color="#CBD5E1",
            height=max(320, len(ordered_for_heat) * 36 + 100),
            margin=dict(l=10, r=60, t=50, b=60),
            xaxis=dict(tickangle=-40, tickfont_size=9, showgrid=False,
                       side='bottom'),
            yaxis=dict(tickfont_size=11, autorange='reversed', showgrid=False),
        )

        # Inline legend
        fig_heat.add_annotation(
            x=0, y=1.08, xref='paper', yref='paper', showarrow=False,
            xanchor='left',
            text=(
                "<span style='color:#1D4ED8'>■</span> On chart   "
                "<span style='color:#DB2777'>■</span> Comeback month   "
                "<span style='color:#374151'>■</span> Off chart   "
                "  Cell = peak rank that month   "
                "<span style='color:#EC4899'>N×</span> = total comebacks"
            ),
            font=dict(size=10, color='#94A3B8'),
            align='left',
        )

        st.plotly_chart(fig_heat, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("🔵 Blue = on chart  ·  🩷 Pink = comeback month  ·  "
                   "Cell shows peak chart rank  ·  Right number = total comeback count")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    # Re-entry gap distribution
    st.markdown('<p class="sec-head">📊 Gap Length Distribution (Days Off-Chart Before Comeback)</p>',
                unsafe_allow_html=True)
    gap_data = df[df['is_reentry_day']]['gap_days']
    fig_gap = px.histogram(
        gap_data, nbins=40,
        labels={"value": "Days Off-Chart", "count": "Number of Re-Entries"},
        color_discrete_sequence=[C_PURPLE],
    )
    fig_gap.update_traces(marker_line_width=0.3, marker_line_color="#0F172A")
    styled_fig(fig_gap, height=280)
    fig_gap.update_layout(showlegend=False)
    st.plotly_chart(fig_gap, use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(f"Median gap: {int(gap_data.median())} days · Mean gap: {gap_data.mean():.0f} days · "
               f"Max gap: {int(gap_data.max())} days")


# ══════════════════════════════════════════════════════════════
#  TAB 2 — MOMENTUM SPIKE DETECTION
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Momentum Spike Detection")
    st.markdown("Popularity trajectory and rank movement for selected songs. "
                "Pink markers = comeback (re-entry) days.")

    # Song picker
    song_options = ss.nlargest(50, 'max_momentum_spike')['song_key'].tolist()
    default_pick = song_options[:3] if len(song_options) >= 3 else song_options
    sel_songs = st.multiselect(
        "Select songs to inspect (max 5 recommended):",
        options=sorted(df['song_key'].unique()),
        default=default_pick,
        format_func=lambda x: x.replace(' || ', ' — '),
    )
    if len(sel_songs) > 5:
        st.warning("Showing too many songs makes these charts unreadable. "
                   "Select 5 or fewer for best clarity.")

    if not sel_songs:
        st.info("Select at least one song above to view momentum charts.")
    else:
        plot_df    = df[df['song_key'].isin(sel_songs)].copy()
        reentry_df = plot_df[plot_df['is_reentry_day']].copy()

        # Short display labels for legend — just song name, max 20 chars
        label_map  = {k: k.split(' || ')[0][:22] for k in sel_songs}
        plot_df['song_label']    = plot_df['song_key'].map(label_map)
        reentry_df['song_label'] = reentry_df['song_key'].map(label_map)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<p class="sec-head">📈 Popularity Over Time</p>',
                        unsafe_allow_html=True)
            fig_pop = px.line(
                plot_df, x='date', y='popularity',
                color='song_label',
                labels={'popularity': 'Popularity Score', 'date': 'Date',
                        'song_label': 'Song'},
                color_discrete_sequence=STINT_COLORS,
            )
            # Add comeback markers — ONE combined trace
            if not reentry_df.empty:
                fig_pop.add_trace(go.Scatter(
                    x=reentry_df['date'],
                    y=reentry_df['popularity'],
                    mode='markers',
                    marker=dict(size=9, color=C_PINK, symbol='star',
                                line=dict(width=1, color='white')),
                    name='Comeback Day',
                    showlegend=True,
                    hovertemplate="<b>Comeback!</b><br>%{text}<br>Pop: %{y}<extra></extra>",
                    text=reentry_df['song_label'],
                ))
            fig_pop.update_layout(**PLOTLY_BASE, height=380,
                                   legend=dict(bgcolor="rgba(15,23,42,0.9)",
                                               bordercolor="#334155", borderwidth=1,
                                               title="Song", font_size=11,
                                               x=1, xanchor="left"))
            fig_pop.update_xaxes(gridcolor="#1E293B")
            fig_pop.update_yaxes(gridcolor="#1E293B")
            st.plotly_chart(fig_pop, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("⭐ Pink stars = comeback (re-entry) days")

        with c2:
            st.markdown('<p class="sec-head">📉 Chart Rank Over Time (lower = better)</p>',
                        unsafe_allow_html=True)
            fig_rank = px.line(
                plot_df, x='date', y='position',
                color='song_label',
                labels={'position': 'Chart Position', 'date': 'Date',
                        'song_label': 'Song'},
                color_discrete_sequence=STINT_COLORS,
            )
            fig_rank.update_yaxes(autorange='reversed', title='Chart Position (1=Top)')
            if not reentry_df.empty:
                fig_rank.add_trace(go.Scatter(
                    x=reentry_df['date'],
                    y=reentry_df['position'],
                    mode='markers',
                    marker=dict(size=9, color=C_PINK, symbol='star',
                                line=dict(width=1, color='white')),
                    name='Comeback Day',
                    showlegend=True,
                    hovertemplate="<b>Comeback!</b><br>Rank: #%{y}<br>%{text}<extra></extra>",
                    text=reentry_df['song_label'],
                ))
            fig_rank.update_layout(**PLOTLY_BASE, height=380,
                                    legend=dict(bgcolor="rgba(15,23,42,0.9)",
                                                bordercolor="#334155", borderwidth=1,
                                                title="Song", font_size=11,
                                                x=1, xanchor="left"))
            fig_rank.update_xaxes(gridcolor="#1E293B")
            fig_rank.update_yaxes(gridcolor="#1E293B")
            st.plotly_chart(fig_rank, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("⭐ Pink stars = comeback days · Y-axis inverted: #1 is top")

        st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

        # Momentum spike score bar
        st.markdown('<p class="sec-head">⚡ Top Comeback Momentum Spikes</p>',
                    unsafe_allow_html=True)
        spike_df = (df[df['is_reentry_day'] & df['song_key'].isin(sel_songs)]
                    .nlargest(20, 'momentum_spike_score')
                    [['song','artist','date','position','popularity',
                      'momentum_spike_score','re_entry_number']]
                    .copy())
        spike_df['label'] = spike_df['song'].str[:25] + ' #' + spike_df['re_entry_number'].astype(str)
        spike_df['date_str'] = spike_df['date'].dt.strftime('%d %b %Y')

        fig_spike = px.bar(
            spike_df.sort_values('momentum_spike_score'),
            x='momentum_spike_score', y='label',
            orientation='h',
            color='momentum_spike_score',
            color_continuous_scale=[[0, '#1E293B'], [0.5, C_PURPLE], [1, C_PINK]],
            text='momentum_spike_score',
            hover_data={'date_str': True, 'position': True, 'popularity': True},
            labels={'momentum_spike_score': 'Momentum Spike Score',
                    'label': '', 'date_str': 'Date', 'position': 'Chart Position',
                    'popularity': 'Popularity'},
        )
        fig_spike.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                                textfont_size=10)
        fig_spike.update_layout(**PLOTLY_BASE, height=max(300, len(spike_df)*28),
                                 coloraxis_showscale=False, showlegend=False,
                                 xaxis=dict(showgrid=False, visible=False))
        fig_spike.update_yaxes(tickfont_size=11)
        st.plotly_chart(fig_spike, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Momentum Spike Score = Popularity jump × Chart position quality on comeback day")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — COMEBACK vs FIRST ENTRY COMPARISON
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Comeback vs First Entry Comparison")
    st.markdown("Do songs perform better on comebacks than their original chart debut?")

    # Build per-stint aggregates
    stint_agg = (df.groupby(['song_key','song','artist','album_type','stint'])
                 .agg(peak_rank=('position','min'),
                      avg_pop=('popularity','mean'),
                      days_on=('date','count'),
                      stint_type=('re_entry_number', lambda x: 'Re-Entry' if x.max()>0 else 'First Entry'))
                 .reset_index())
    stint_agg['stint_label'] = stint_agg['stint'].apply(
        lambda x: 'First Entry' if x == 0 else f'Comeback #{x}')

    # Aggregate first entry vs all comebacks
    entry_compare = stint_agg.groupby('stint_type').agg(
        avg_peak_rank=('peak_rank','mean'),
        avg_popularity=('avg_pop','mean'),
        avg_days=('days_on','mean'),
        count=('song_key','count'),
    ).reset_index()
    entry_compare = entry_compare[entry_compare['stint_type'].isin(['First Entry','Re-Entry'])]

    c1, c2, c3 = st.columns(3, gap="large")
    metrics = [
        (c1, 'avg_peak_rank',   'Avg Peak Rank',        '(lower is better)', True),
        (c2, 'avg_popularity',  'Avg Popularity Score', '(higher is better)', False),
        (c3, 'avg_days',        'Avg Days on Chart',    '(per stint)',        False),
    ]
    for col, metric, title, note, invert in metrics:
        with col:
            st.markdown(f'<p class="sec-head">{title}<br><span style="color:#475569;font-size:.68rem">{note}</span></p>',
                        unsafe_allow_html=True)
            fe = entry_compare[entry_compare['stint_type']=='First Entry'][metric].values
            re = entry_compare[entry_compare['stint_type']=='Re-Entry'][metric].values
            vals = [float(fe[0]) if len(fe) else 0, float(re[0]) if len(re) else 0]
            fig = go.Figure(go.Bar(
                x=['First Entry', 'Re-Entry (Comeback)'],
                y=vals,
                marker_color=[C_BLUE, C_PINK],
                text=[f'{v:.1f}' for v in vals],
                textposition='outside',
                textfont_size=13,
            ))
            fig.update_layout(**PLOTLY_BASE, height=280, showlegend=False,
                              yaxis=dict(autorange='reversed' if invert else True,
                                         gridcolor='#1E293B'))
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    # Comeback performance by comeback number
    st.markdown('<p class="sec-head">📈 Performance by Comeback Number (1st, 2nd, 3rd comeback…)</p>',
                unsafe_allow_html=True)
    comeback_perf = (stint_agg[stint_agg['stint'] > 0]
                     .groupby('stint').agg(
                         avg_peak_rank=('peak_rank','mean'),
                         avg_pop=('avg_pop','mean'),
                         count=('song_key','count'))
                     .reset_index()
                     .query('stint <= 10'))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_cr = px.line(comeback_perf, x='stint', y='avg_peak_rank',
                         markers=True,
                         labels={'stint': 'Comeback Number', 'avg_peak_rank': 'Avg Peak Rank'},
                         color_discrete_sequence=[C_PINK])
        fig_cr.update_yaxes(autorange='reversed', title='Avg Peak Rank (lower=better)')
        styled_fig(fig_cr, height=300)
        fig_cr.update_layout(showlegend=False)
        st.plotly_chart(fig_cr, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Does rank improve or decline with each successive comeback?")

    with c2:
        fig_cp = px.line(comeback_perf, x='stint', y='avg_pop',
                         markers=True,
                         labels={'stint': 'Comeback Number', 'avg_pop': 'Avg Popularity'},
                         color_discrete_sequence=[C_PURPLE])
        styled_fig(fig_cp, height=300)
        fig_cp.update_layout(showlegend=False)
        st.plotly_chart(fig_cp, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Popularity trajectory across multiple comebacks")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    # Single vs album comeback comparison
    st.markdown('<p class="sec-head">💿 Singles vs Albums — Comeback Strength</p>',
                unsafe_allow_html=True)
    type_compare = (stint_agg[stint_agg['stint'] > 0]
                    .groupby('album_type').agg(
                        avg_peak_rank=('peak_rank','mean'),
                        avg_pop=('avg_pop','mean'),
                        avg_days=('days_on','mean'),
                        count=('song_key','nunique'))
                    .reset_index())

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_tc = px.bar(type_compare, x='album_type', y='avg_peak_rank',
                        color='album_type',
                        color_discrete_map=ALBUM_COLORS,
                        text='avg_peak_rank',
                        labels={'album_type': 'Release Type', 'avg_peak_rank': 'Avg Peak Rank'},
                        title='Avg Peak Rank on Comeback (lower = better)')
        fig_tc.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_tc.update_yaxes(autorange='reversed')
        styled_fig(fig_tc, height=300)
        fig_tc.update_layout(showlegend=False)
        st.plotly_chart(fig_tc, use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        fig_td = px.bar(type_compare, x='album_type', y='avg_days',
                        color='album_type',
                        color_discrete_map=ALBUM_COLORS,
                        text='avg_days',
                        labels={'album_type': 'Release Type', 'avg_days': 'Avg Days per Stint'},
                        title='Avg Days on Chart per Comeback Stint')
        fig_td.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        styled_fig(fig_td, height=300)
        fig_td.update_layout(showlegend=False)
        st.plotly_chart(fig_td, use_container_width=True,
                        config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
#  TAB 4 — CONTENT ATTRIBUTES vs MOMENTUM
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Content Attribute vs Momentum Analysis")
    st.markdown("How do song characteristics — duration, explicitness, album size, release type — "
                "relate to comeback strength?")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="sec-head">⏱ Song Duration vs Momentum Spike</p>',
                    unsafe_allow_html=True)
        dur_data = ss[ss['total_reentries'] > 0].copy()
        dur_data['duration_bucket'] = pd.cut(
            dur_data['duration_min'],
            bins=[0, 2, 3, 4, 5, 20],
            labels=['< 2 min', '2–3 min', '3–4 min', '4–5 min', '5+ min']
        )
        dur_bin = (dur_data.groupby('duration_bucket', observed=True).agg(
            Avg_Momentum   = ('max_momentum_spike', 'mean'),
            Avg_Reentries  = ('total_reentries', 'mean'),
            Song_Count     = ('song_key', 'count'),
        ).reset_index().rename(columns={'duration_bucket': 'Duration Range'}))

        fig_dur = go.Figure()
        fig_dur.add_trace(go.Bar(
            x=dur_bin['Duration Range'], y=dur_bin['Avg_Momentum'],
            name='Avg Momentum Spike', marker_color=C_PURPLE,
            text=dur_bin['Avg_Momentum'].round(1),
            textposition='outside', textfont_size=11,
        ))
        fig_dur.add_trace(go.Scatter(
            x=dur_bin['Duration Range'], y=dur_bin['Song_Count'],
            name='Song Count', mode='lines+markers',
            marker=dict(color=C_AMBER, size=8),
            line=dict(color=C_AMBER, dash='dot'),
            yaxis='y2',
        ))
        fig_dur.update_layout(
            **PLOTLY_BASE, height=360,
            yaxis=dict(title='Avg Momentum Spike', gridcolor='#1E293B'),
            yaxis2=dict(title='Song Count', overlaying='y', side='right',
                        showgrid=False, tickfont_color=C_AMBER),
            legend=dict(bgcolor="rgba(15,23,42,0.9)", bordercolor="#334155",
                        borderwidth=1, font_size=11),
            xaxis=dict(title='Duration Range', gridcolor='#1E293B'),
            bargap=0.3,
        )
        st.plotly_chart(fig_dur, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("🟣 Bar = avg momentum spike per duration bucket  |  🟡 Line = number of songs in that bucket")

    with c2:
        st.markdown('<p class="sec-head">🔞 Explicit vs Clean — Comeback Momentum</p>',
                    unsafe_allow_html=True)
        exp_compare = (ss.groupby('is_explicit').agg(
            avg_momentum=('max_momentum_spike','mean'),
            avg_reentries=('total_reentries','mean'),
            avg_popularity=('avg_popularity','mean'),
            count=('song_key','count'),
        ).reset_index())
        exp_compare['label'] = exp_compare['is_explicit'].map(
            {True: '🔞 Explicit', False: '✅ Clean'})

        fig_exp = px.bar(
            exp_compare, x='label',
            y=['avg_momentum','avg_reentries','avg_popularity'],
            barmode='group',
            color_discrete_map={
                'avg_momentum': C_PURPLE,
                'avg_reentries': C_PINK,
                'avg_popularity': C_TEAL,
            },
            labels={'label': '', 'value': 'Score', 'variable': 'Metric'},
        )
        # Clean up legend names
        name_map = {'avg_momentum': 'Avg Momentum Spike',
                    'avg_reentries': 'Avg Re-Entries',
                    'avg_popularity': 'Avg Popularity'}
        for trace in fig_exp.data:
            trace.name = name_map.get(trace.name, trace.name)
        styled_fig(fig_exp, height=360)
        st.plotly_chart(fig_exp, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(f"Explicit songs: {exp_compare[exp_compare['is_explicit']]['count'].values[0] if True in exp_compare['is_explicit'].values else 0} · "
                   f"Clean songs: {exp_compare[~exp_compare['is_explicit']]['count'].values[0] if False in exp_compare['is_explicit'].values else 0}")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="sec-head">📀 Album Size vs Re-Entry Frequency</p>',
                    unsafe_allow_html=True)
        alb_data = ss[ss['total_reentries'] > 0].copy()
        alb_data['track_bucket'] = pd.cut(
            alb_data['total_tracks'],
            bins=[0, 1, 5, 10, 20, 200],
            labels=['1 track\n(Single)', '2–5\ntracks', '6–10\ntracks',
                    '11–20\ntracks', '20+\ntracks']
        )
        alb_bin = (alb_data.groupby('track_bucket', observed=True).agg(
            Avg_Reentries  = ('total_reentries', 'mean'),
            Avg_Momentum   = ('max_momentum_spike', 'mean'),
            Song_Count     = ('song_key', 'count'),
        ).reset_index().rename(columns={'track_bucket': 'Album Size'}))

        fig_alb = go.Figure()
        fig_alb.add_trace(go.Bar(
            x=alb_bin['Album Size'], y=alb_bin['Avg_Reentries'],
            name='Avg Re-Entries', marker_color=C_TEAL,
            text=alb_bin['Avg_Reentries'].round(1),
            textposition='outside', textfont_size=11,
        ))
        fig_alb.add_trace(go.Bar(
            x=alb_bin['Album Size'], y=alb_bin['Avg_Momentum'],
            name='Avg Momentum Spike', marker_color=C_PURPLE,
            text=alb_bin['Avg_Momentum'].round(1),
            textposition='outside', textfont_size=11,
        ))
        fig_alb.update_layout(
            **PLOTLY_BASE, height=360, barmode='group', bargap=0.2,
            yaxis=dict(title='Score', gridcolor='#1E293B'),
            xaxis=dict(title='Album Track Count Range', gridcolor='#1E293B'),
            legend=dict(bgcolor="rgba(15,23,42,0.9)", bordercolor="#334155",
                        borderwidth=1, font_size=11),
        )
        st.plotly_chart(fig_alb, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("🩵 Avg re-entries  🟣 Avg momentum spike — grouped by how many tracks are in the release")

    with c2:
        st.markdown('<p class="sec-head">💿 Album Type — Fandom Intensity Distribution</p>',
                    unsafe_allow_html=True)
        fig_box = px.box(
            ss, x='album_type', y='fandom_intensity_score',
            color='album_type',
            color_discrete_map=ALBUM_COLORS,
            points='all',
            hover_data=['song','artist'],
            labels={'album_type': 'Release Type',
                    'fandom_intensity_score': 'Fandom Intensity Score (0–100)'},
        )
        fig_box.update_traces(marker_size=4)
        styled_fig(fig_box, height=340)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Box shows median ± IQR · Dots = individual songs")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    # Heatmap: Artist × Comeback metric
    st.markdown('<p class="sec-head">🔥 Artist Comeback Heatmap — Avg Momentum by Artist</p>',
                unsafe_allow_html=True)
    artist_heat = (df[df['is_reentry_day']]
                   .groupby('artist')['momentum_spike_score']
                   .agg(['mean','count'])
                   .reset_index()
                   .query('count >= 3')
                   .nlargest(20,'mean')
                   .rename(columns={'mean':'Avg Spike','count':'Comebacks'}))

    fig_heat = px.bar(
        artist_heat.sort_values('Avg Spike'),
        x='Avg Spike', y='artist',
        orientation='h',
        color='Avg Spike',
        color_continuous_scale=[[0,'#1E293B'],[0.5, C_PURPLE],[1, C_PINK]],
        text='Avg Spike',
        hover_data={'Comebacks': True},
        labels={'artist': '', 'Avg Spike': 'Avg Momentum Spike Score'},
    )
    fig_heat.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                           textfont_size=10)
    fig_heat.update_layout(**PLOTLY_BASE, height=480, showlegend=False,
                            coloraxis_showscale=False,
                            xaxis=dict(showgrid=False, visible=False))
    fig_heat.update_yaxes(tickfont_size=11)
    st.plotly_chart(fig_heat, use_container_width=True,
                    config={"displayModeBar": False})
    st.caption("Only artists with ≥3 comeback events shown")


# ══════════════════════════════════════════════════════════════
#  TAB 5 — FANDOM INTENSITY LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Fandom Intensity Leaderboard")
    st.markdown("Composite score combining re-entry frequency (40%), momentum spike strength (35%), "
                "and average popularity (25%). Higher = stronger fandom-driven re-activation.")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        lb_type = st.selectbox("Album type", ["All","single","album"], index=0)
    with c2:
        lb_explicit = st.selectbox("Explicit filter", ["All","Clean only","Explicit only"], index=0)
    with c3:
        lb_min_re = st.number_input("Min re-entries", 0, 30, 1)

    lb_df = ss.copy()
    if lb_type != "All":
        lb_df = lb_df[lb_df['album_type'] == lb_type]
    if lb_explicit == "Clean only":
        lb_df = lb_df[~lb_df['is_explicit']]
    elif lb_explicit == "Explicit only":
        lb_df = lb_df[lb_df['is_explicit']]
    lb_df = lb_df[lb_df['total_reentries'] >= lb_min_re]
    lb_df = lb_df.nlargest(50, 'fandom_intensity_score').reset_index(drop=True)
    lb_df.index += 1

    # Top 3 callout
    if len(lb_df) >= 3:
        cols = st.columns(3)
        medals = ["🥇","🥈","🥉"]
        for i, col in enumerate(cols):
            row = lb_df.iloc[i]
            with col:
                st.markdown(f"""
                <div class="kpi-wrap">
                    <div class="kpi-label">{medals[i]} Rank {i+1}</div>
                    <div class="kpi-value" style="font-size:1.1rem;">{row['song'][:22]}</div>
                    <div class="kpi-sub">{row['artist']} · Score: {row['fandom_intensity_score']}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard chart
    fig_lb = px.bar(
        lb_df.head(25).sort_values('fandom_intensity_score'),
        x='fandom_intensity_score',
        y=lb_df.head(25).sort_values('fandom_intensity_score').apply(
            lambda r: f"{r['song'][:28]} — {r['artist'][:18]}", axis=1),
        orientation='h',
        color='album_type',
        color_discrete_map=ALBUM_COLORS,
        text='fandom_intensity_score',
        hover_data={'total_reentries': True, 'avg_popularity': True,
                    'peak_rank': True, 'fandom_intensity_score': True},
        labels={'fandom_intensity_score': 'Fandom Intensity Score (0–100)',
                'y': '', 'album_type': 'Release Type',
                'total_reentries': 'Re-Entries',
                'avg_popularity': 'Avg Popularity',
                'peak_rank': 'Peak Rank'},
    )
    fig_lb.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                         textfont_size=9)
    fig_lb.update_layout(**PLOTLY_BASE,
                          height=max(400, len(lb_df.head(25)) * 26),
                          xaxis=dict(range=[0,115], showgrid=False, visible=False),
                          legend=dict(title="Release Type",
                                      bgcolor="rgba(15,23,42,0.9)",
                                      bordercolor="#334155", borderwidth=1))
    fig_lb.update_yaxes(tickfont_size=10)
    st.plotly_chart(fig_lb, use_container_width=True,
                    config={"displayModeBar": False})
    st.caption("🟣 Single  |  🩵 Album  |  Score = Re-Entry Freq (40%) + Momentum Spike (35%) + Popularity (25%)")

    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

    # Full sortable table
    st.markdown('<p class="sec-head">📋 Full Leaderboard Table</p>', unsafe_allow_html=True)
    table_df = lb_df[['song','artist','album_type','is_explicit',
                       'total_reentries','max_momentum_spike','avg_popularity',
                       'peak_rank','avg_days_per_stint','fandom_intensity_score']].copy()
    table_df.columns = ['Song','Artist','Type','Explicit','Re-Entries',
                         'Peak Spike','Avg Pop','Peak Rank','Avg Stint Days','Fandom Score']
    table_df['Explicit'] = table_df['Explicit'].map({True:'🔞', False:'✅'})

    def color_score(val):
        if val >= 50: return 'background-color:#1a1033; color:#A78BFA'
        if val >= 25: return 'background-color:#0d1f2d; color:#60A5FA'
        return ''

    styled = (table_df.style
              .map(color_score, subset=['Fandom Score'])
              .format({'Peak Spike':'{:.1f}','Avg Pop':'{:.1f}',
                       'Avg Stint Days':'{:.1f}','Fandom Score':'{:.1f}'}))
    st.dataframe(styled, height=480, use_container_width=True, hide_index=False)
    st.markdown('<p class="sec-head">💡 Atlantic Action Points</p>',
                unsafe_allow_html=True)
    st.markdown("""
- **Songs scoring above 50** have demonstrated genuine fandom re-activation — prioritise these artists for dedicated comeback campaign budgets
- **High spike + low re-entry count** songs are underutilised — a structured promotional push could trigger additional chart runs
- **Singles consistently outperform albums** on comeback retention — recommend single-first release strategies for K-Pop signings
- **Clean content dominates** South Korea's Top 50 — explicit tracks underperform on re-entry frequency and average popularity
- **Artists with comeback spikes above 20** (see Tab 4) should be fast-tracked for collaborative promotions and cross-market exposure
    """)
