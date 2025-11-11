# streamlit_app_final.py
# GetAround — Delay analysis

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

plotly_cfg_png_only = {
    "displaylogo": False,
    "modeBarButtons": [["toImage"]],  # show only the download button
    "toImageButtonOptions": {"format": "png"}  # force PNG
}

# --- Card template (icon on top, centered) ---
def metric_card(title, value, emoji):
    if isinstance(value, (int, float, np.integer, np.floating)):
        display_value = f"{value:,}"
    else:
        display_value = "" if value is None else str(value)
    return f"""
    <div style="
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="font-size:26px; line-height:1; margin-bottom:6px;">{emoji}</div>
        <div style="font-weight:600; margin-bottom:6px;">{title}</div>
        <div style="font-size:28px; margin:0;">{display_value}</div>
    </div>
    """

st.set_page_config(page_title="GetAround — Delay Decision Helper", layout="centered")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
data_path = BASE_DIR / "data" / "processed" / "getaround_delay_for_streamlit.csv"
df = pd.read_csv(data_path)
#st.write("Data loaded:", df.shape)

# ----------------------------
# Header
# ----------------------------
st.title("GetAround Dashboard")

st.markdown("""
    Data analysis on car rental late returns. A Jedha project!
""")

st.markdown("""
<style>
/* Target the tablist buttons with extra specificity */
div[data-testid="stTabs"] > div[role="tablist"] > button {
  font-size: 1.5rem !important;      /* bigger */
  line-height: 1.4 !important;       /* better vertical spacing */
  font-weight: 700 !important;
  padding: 0.6rem 1rem !important;
}

/* Some Streamlit builds wrap text in p/span; bump those too */
div[data-testid="stTabs"] > div[role="tablist"] > button p,
div[data-testid="stTabs"] > div[role="tablist"] > button span {
  font-size: 1.5rem !important;
  line-height: 1.4 !important;
  font-weight: 700 !important;
}

/* Active tab highlight */
div[data-testid="stTabs"] > div[role="tablist"] > button[aria-selected="true"] {
  border-bottom: 3px solid #0F4C81 !important;
  color: #0F4C81 !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
:root, body, [data-testid="stMarkdownContainer"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
               Ubuntu, Cantarell, "Helvetica Neue", Arial, "Apple Color Emoji",
               "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Analyse descriptive", "🎯 Threshold picker"])


with tab1:
    st.markdown("""
        ## KPI
    """)

    # --- Compute metrics ---
    n_cars      = df["car_id"].nunique()
    n_rentals   = df["rental_id"].nunique()
    n_canceled  = (df["state"] == "canceled").sum()
    n_late      = df["delay_at_checkout_in_minutes"].gt(0).sum()  # current late returns
    n_connect   = (df["checkin_type"] == "connect").sum()
        # ---------- BASE (no scope) for STATIC CARDS ----------
    base_all = df.loc[
        df["previous_ended_rental_id"].notna() &
        df["previous_delay_at_checkout_in_minutes"].notna() &
        df["time_delta_with_previous_rental_in_minutes"].notna(),
        ["state", "checkin_type",
         "previous_delay_at_checkout_in_minutes",
         "time_delta_with_previous_rental_in_minutes"]
    ].rename(columns={
        "previous_delay_at_checkout_in_minutes": "prev_delay",
        "time_delta_with_previous_rental_in_minutes": "gap"
    })
    n_pairs_all = len(base_all)
    impacted_before_rate_all = (base_all["prev_delay"] > base_all["gap"]).mean()

    base_all["is_canceled"] = base_all["state"].eq("canceled")
    base_all["due_to_prev_late"] = base_all["is_canceled"] & (base_all["prev_delay"] > base_all["gap"])
    n_canceled_all = int(base_all["is_canceled"].sum())
    n_canceled_due_prev_all = int(base_all["due_to_prev_late"].sum())
    rate_canceled_pairs_all = n_canceled_all / n_pairs_all if n_pairs_all else 0.0
    rate_canceled_due_pairs_all = n_canceled_due_prev_all / n_pairs_all if n_pairs_all else 0.0
    share_due_within_canceled_all = (n_canceled_due_prev_all / n_canceled_all) if n_canceled_all else 0.0

    # --- 4 cards on top ---
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Unique cars", n_cars, "🚗"), unsafe_allow_html=True)
    c2.markdown(metric_card("Unique rentals", n_rentals, "🧾"), unsafe_allow_html=True)
    c3.markdown(metric_card("Cancellations", n_canceled, "❌"), unsafe_allow_html=True)
    c4.markdown(metric_card("Late returns", n_late, "⌛"), unsafe_allow_html=True)

    # --- 4 cards underneath ---   
    c5, c6, c7, c8 = st.columns(4)    
    c5.markdown(metric_card("Pairs considered (overall)", f"{n_pairs_all:,}", "🔗"), unsafe_allow_html=True)
    c6.markdown(metric_card("Pairs considered (overall)", f"{n_pairs_all:,}", "🔗"), unsafe_allow_html=True)
    c7.markdown(metric_card("Pairs considered (overall)", f"{n_pairs_all:,}", "🔗"), unsafe_allow_html=True)
    c8.markdown(metric_card("Pairs considered (overall)", f"{n_pairs_all:,}", "🔗"), unsafe_allow_html=True)

    st.divider()

    # --- Pie chart (no card, centered title) ---
    st.markdown('<h4 style="text-align:center;margin:0 0 6px 0;">⌛ Proportion of cancellations</h4>', unsafe_allow_html=True)

    pie_df = (
        df["state"]
        .map({"ended": "Ended", "canceled": "Canceled"})
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="count")
    )

    fig = px.pie(pie_df, values="count", names="Status", hole=0.35)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))

    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg_png_only)
    st.divider()

    # ----------------------------
    # Late returns
    # ----------------------------
    late_mask = df["delay_at_checkout_in_minutes"].gt(0).fillna(False)
    late_rate = 100 * late_mask.mean()
    n_late = int(late_mask.sum())

    # Title with % late
    st.markdown(
        f'<h4 style="text-align:center;margin:0 0 6px 0;">⌛ Late returns: {late_rate:.1f}% — distribution by check-in type</h4>',
        unsafe_allow_html=True
    )

    # Proportion of checkin_type among late returns
    late_by_checkin = (
        df.loc[late_mask, "checkin_type"]
        .dropna()
        .value_counts(normalize=True)
        .rename_axis("checkin_type")
        .reset_index(name="share")
    )

    # Simple bar chart (proportions)
    fig_late = px.bar(late_by_checkin, x="checkin_type", y="share", text="share")
    fig_late.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig_late.update_yaxes(tickformat=".0%", range=[0, 1])
    fig_late.update_layout(
        xaxis_title="Check-in type",
        yaxis_title="Share of late returns",
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
    )

    st.plotly_chart(fig_late, use_container_width=True, config=plotly_cfg_png_only)

    st.divider()


    # ----------------------------
    # Gaps by check-in type (comparison)
    # ----------------------------
    st.markdown("### 🔄 Gaps between rentals by check-in type")

    df_gap_type = (
        df.loc[(df["previous_ended_rental_id"].notna()) & (df["state"] == "ended"),
            ["checkin_type", "time_delta_with_previous_rental_in_minutes"]]
        .dropna()
        .rename(columns={"time_delta_with_previous_rental_in_minutes": "gap_minutes"})
    )

    # Count per gap per type
    df_line_type = (
        df_gap_type.groupby(["checkin_type", "gap_minutes"])
                .size()
                .reset_index(name="count")
                .sort_values(["checkin_type", "gap_minutes"])
    )

    fig_gap_type = px.line(
        df_line_type, 
        x="gap_minutes", 
        y="count", 
        color="checkin_type",
        color_discrete_map={"connect": "green", "mobile": "blue"},
        title="Number of rentals by gap duration and check-in type (0–750 min view)"
    )

    fig_gap_type.update_traces(mode="lines", line=dict(width=2))
    fig_gap_type.update_xaxes(title="Gap between rentals (minutes)", range=[0, 750])
    fig_gap_type.update_yaxes(title="Number of rentals")
    fig_gap_type.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
        legend_title_text="Check-in type"
    )

    st.plotly_chart(fig_gap_type, use_container_width=True, config=plotly_cfg_png_only)

    st.divider()

    # ----------------------------
    # Impact on next driver (no buffer yet)
    # ----------------------------
    st.markdown("## Impact on next driver (current situation)")

    df_impact = df.loc[
        (df["previous_ended_rental_id"].notna()) &
        (df["state"] == "ended") &
        df["previous_delay_at_checkout_in_minutes"].notna() &
        df["time_delta_with_previous_rental_in_minutes"].notna(),
        ["checkin_type", "previous_delay_at_checkout_in_minutes", "time_delta_with_previous_rental_in_minutes"]
    ].copy()

    # Determine whether next driver was impacted (no threshold yet)
    df_impact["next_driver_impacted"] = (
        df_impact["previous_delay_at_checkout_in_minutes"] >
        df_impact["time_delta_with_previous_rental_in_minutes"]
    )

    # Compute count and percentage by check-in type
    impact_summary = (
        df_impact.groupby("checkin_type")["next_driver_impacted"]
        .agg(["sum", "count"])
        .reset_index()
        .assign(impact_rate=lambda d: d["sum"] / d["count"])
    )

    # Plot bar chart
    fig_impact = px.bar(
        impact_summary,
        x="checkin_type",
        y="impact_rate",
        text="impact_rate",
        title="Share of next rentals impacted (no buffer threshold)",
    )
    fig_impact.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig_impact.update_yaxes(title="Impact rate", tickformat=".0%")
    fig_impact.update_xaxes(title="Check-in type")
    fig_impact.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=370)

    st.plotly_chart(fig_impact, use_container_width=True, config=plotly_cfg_png_only)
    st.divider()

with tab2:

    # ----------------------------
    # Buffer threshold — impact vs. blocked rentals
    # ----------------------------
    st.markdown("## Buffer threshold — impact vs. blocked rentals")

    # ---------- BASE (no scope) for STATIC CARDS ----------
    base_all = df.loc[
        df["previous_ended_rental_id"].notna() &
        df["previous_delay_at_checkout_in_minutes"].notna() &
        df["time_delta_with_previous_rental_in_minutes"].notna(),
        ["state", "checkin_type",
         "previous_delay_at_checkout_in_minutes",
         "time_delta_with_previous_rental_in_minutes"]
    ].rename(columns={
        "previous_delay_at_checkout_in_minutes": "prev_delay",
        "time_delta_with_previous_rental_in_minutes": "gap"
    })

    impacted_before_rate_all = (base_all["prev_delay"] > base_all["gap"]).mean()

    base_all["is_canceled"] = base_all["state"].eq("canceled")
    base_all["due_to_prev_late"] = base_all["is_canceled"] & (base_all["prev_delay"] > base_all["gap"])
    n_canceled_all = int(base_all["is_canceled"].sum())
    n_canceled_due_prev_all = int(base_all["due_to_prev_late"].sum())
    rate_canceled_pairs_all = n_canceled_all / n_pairs_all if n_pairs_all else 0.0
    rate_canceled_due_pairs_all = n_canceled_due_prev_all / n_pairs_all if n_pairs_all else 0.0
    share_due_within_canceled_all = (n_canceled_due_prev_all / n_canceled_all) if n_canceled_all else 0.0

    # ---------- STATIC CARDS (overall, no scope) ----------
    c1, c2, c3, c4 = st.columns(4)

    c2.markdown(metric_card("Impacted (no buffer, overall)", f"{impacted_before_rate_all:.1%}", "⚠️"), unsafe_allow_html=True)
    c3.markdown(metric_card("Canceled (overall)", f"{rate_canceled_pairs_all:.1%}", "❌"), unsafe_allow_html=True)
    c4.markdown(metric_card("Canceled due to prev delay (overall)", f"{rate_canceled_due_pairs_all:.1%}", "⏱️"), unsafe_allow_html=True)
    st.caption(f"Within cancellations: {share_due_within_canceled_all:.1%} due to previous lateness (overall).")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.divider()

    # ---------- CHART PLACEHOLDER (will render above controls) ----------
    chart_area = st.empty()

    # ---------- CONTROLS (scope just above the slider) ----------
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        include_mobile = st.checkbox("Mobile", value=True, key="t2_mobile")
    with colB:
        include_connect = st.checkbox("Connect", value=True, key="t2_connect")
    with colC:
        show_pct = st.checkbox("Show percentage", value=True, key="t2_pct")

    threshold_slider = st.slider(
        "Minimum buffer (minutes)",
        min_value=0, max_value=180, value=60, step=15,
        help="Policy buffer to enforce between two rentals",
        key="t2_slider"
    )

    # ---------- SCOPED DATA (affects chart + result card + summary) ----------
    scope = []
    if include_mobile: scope.append("mobile")
    if include_connect: scope.append("connect")

    if len(scope) == 0:
        st.info("Select at least one check-in type to simulate.")
    else:
        data = base_all[base_all["checkin_type"].isin(scope)].copy()
        n_pairs = len(data)
        if n_pairs == 0:
            st.info("No rentals match the current filters.")
        else:
            # Baseline (no buffer) in scope
            impacted_before_rate = (data["prev_delay"] > data["gap"]).mean()

            # Precompute simulation across thresholds
            thresholds = np.arange(0, 181, 15)
            prev_delay_np = data["prev_delay"].to_numpy()
            gap_np = data["gap"].to_numpy()

            impacted_after_rate = []
            blocked_rate = []
            for t in thresholds:
                eff_gap = np.maximum(gap_np, t)
                impacted_after_rate.append((prev_delay_np > eff_gap).mean())
                blocked_rate.append((gap_np < t).mean())

            sim_df = pd.DataFrame({
                "threshold_min": thresholds,
                "impacted_after": impacted_after_rate,
                "blocked": blocked_rate
            })
            sim_df["impacted_after_cnt"] = (sim_df["impacted_after"] * n_pairs).round().astype(int)
            sim_df["blocked_cnt"]        = (sim_df["blocked"] * n_pairs).round().astype(int)

            # y
            if show_pct:
                y_cols = ["impacted_after", "blocked"]
                y_title = "Share of rentals"; yfmt = ".0%"
            else:
                y_cols = ["impacted_after_cnt", "blocked_cnt"]
                y_title = "Number of rentals"; yfmt = ""

            # Build & render chart
            fig_trade = px.line(
                sim_df.melt(id_vars="threshold_min", value_vars=y_cols,
                            var_name="metric", value_name="value"),
                x="threshold_min", y="value", color="metric",
                labels={"threshold_min": "Buffer (minutes)", "value": y_title, "metric": "Metric"},
                title="Impact on next driver vs. Rentals blocked (by buffer)"
            )
            fig_trade.update_traces(mode="lines", line=dict(width=3))
            fig_trade.for_each_trace(lambda tr: tr.update(
                name="Impacted after buffer" if "impacted" in tr.name else "Blocked by buffer"
            ))
            if show_pct:
                fig_trade.update_yaxes(tickformat=yfmt)
            fig_trade.add_vline(x=threshold_slider, line_dash="dash", line_width=2)
            fig_trade.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)

            chart_area.plotly_chart(fig_trade, use_container_width=True, config=plotly_cfg_png_only)

            # Result card (depends on slider, scoped)
            row_sel = sim_df.loc[sim_df["threshold_min"] == threshold_slider].iloc[0]
            k_res, = st.columns(1)
            if show_pct:
                k_res.markdown(
                    metric_card(f"Impacted @ {threshold_slider} min (in scope)", f"{row_sel['impacted_after']:.1%}", "🛡️"),
                    unsafe_allow_html=True
                )
                st.caption(
                    f"Scope = **{', '.join(scope)}**. Baseline impacted: **{impacted_before_rate:.1%}**. "
                    f"Blocked at {threshold_slider} min: **{row_sel['blocked']:.1%}** (n={n_pairs:,})."
                )
            else:
                k_res.markdown(
                    metric_card(f"Impacted @ {threshold_slider} min (in scope)", f"{int(row_sel['impacted_after_cnt']):,}", "🛡️"),
                    unsafe_allow_html=True
                )
                st.caption(
                    f"Scope = **{', '.join(scope)}**. Baseline impacted: **{int(round(impacted_before_rate*n_pairs)):,}**. "
                    f"Blocked at {threshold_slider} min: **{int(row_sel['blocked_cnt']):,}** (n={n_pairs:,})."
                )

    st.divider()

    st.write(
        "This dashboard was created by Sandra MARGOT, October 2025"
    )
