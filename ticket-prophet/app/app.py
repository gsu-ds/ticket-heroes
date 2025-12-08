import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import joblib

# Page config
st.set_page_config(
    page_title="Ticket Hero | ML Approaches for Event Price Prediction",
    page_icon="🎫",
    layout="wide",
)

# Paths
APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "TicketProphetLogo.png"
MODEL_PATH = APP_DIR / "models" / "rf_model.joblib"


@st.cache_resource
def load_model():
    """Load trained Random Forest model from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.warning(
            f"Could not load trained Random Forest model from {MODEL_PATH}. "
            "Using demo logic instead."
        )
        return None


rf_model = load_model()

# dark theme
st.markdown(
    """
    <style>
    .top-nav-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #d1d5db; /* lighter gray for dark background */
        margin-bottom: 0.8rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        background: #111827;
        color: #f9fafb;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.4rem;
        border: 1px solid #4b5563;
    }
    .pill + .pill {
        margin-left: 0.15rem;
    }
    .pill span.dot {
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 999px;
        background: #f97316;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# top brand text
st.markdown('<div class="top-nav-title">TicketProphet</div>', unsafe_allow_html=True)

# Tabs
tab_dashboard, tab_methodology, tab_aggregator = st.tabs(
    ["Dashboard", "Methodology", "Market Aggregator"]
)


# ============= DASHBOARD =============
with tab_dashboard:
    # Logo + hero
    st.markdown(
        """
        <style>
            .logo-container {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_logo, col_title = st.columns([0.3, 1.7])
    with col_logo:
        if LOGO_PATH.exists():
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=120)
    with col_title:
        st.markdown(
            '<div class="hero-title">Never Overpay for Live Events Again.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">'
            "Data-driven price prediction using Random Forest and Market Aggregation."
            "</div>",
            unsafe_allow_html=True,
        )

    col_hero_left, col_hero_right = st.columns([2, 1])
    with col_hero_right:
        st.write("")
        st.metric("Model Version", "v2.1", "RandomForest")

    st.markdown("---")

    # Live Event Analytics header
    st.header("Live Event Analytics")
    st.write(
        "This dashboard demonstrates our core regression model. "
        "Static features (market strength, sales momentum) and event-level signals "
        "are used to estimate a fair market value versus the current list price."
    )

    # Shared state + events
    if "search_text" not in st.session_state:
        st.session_state["search_text"] = ""

    EVENTS = [
        "Drake: It's All A Blur",
        "Taylor Swift: The Eras Tour",
        "Bad Bunny: Most Wanted Tour",
        "Miami Heat vs Lakers",
    ]

    # Two-column layout: LEFT = inputs, RIGHT = outputs
    col_inputs, col_outputs = st.columns([1.3, 1.7])

    # ----- LEFT COLUMN: SEARCH + INPUTS -----
    with col_inputs:
        st.markdown("### Model Inputs")
        st.caption("v2.1 (Random Forest Regression)")

        # Search bar inside left column
        st.markdown("**Search Events**")
        search_query = st.text_input(
            "Search by artist, team, or event",
            value=st.session_state["search_text"],
            key="search_text",
            placeholder="Try 'Drake', 'Miami Heat', 'Taylor Swift'...",
            label_visibility="collapsed",
        )

        # Filter events based on search text
        if search_query.strip():
            filtered_events = [e for e in EVENTS if search_query.lower() in e.lower()]
        else:
            filtered_events = EVENTS.copy()

        if not filtered_events:
            filtered_events = EVENTS.copy()

        # Event dropdown (filtered)
        event = st.selectbox("Event / Artist", filtered_events)

        # Remaining inputs
        days_until = st.slider(
            "Days Until Event", min_value=0, max_value=180, value=14, step=1
        )

        inventory = st.radio(
            "Inventory Level",
            [
                "High (>2k)",
                "Medium (500–2k)",
                "Low (<500)",
            ],
        )

        st.markdown("**Current Lowest List Price**")
        current_price = st.number_input(
            "Enter lowest listing price ($)",
            min_value=0.0,
            max_value=10000.0,
            value=250.0,
            step=5.0,
        )

        run = st.button("Run Prediction")

    # ----- RIGHT COLUMN: PREDICTION + CHARTS -----
    with col_outputs:
        st.markdown("### Prediction & Recommendation")

        # Fallback demo model if RF isn't available
        def demo_fair_value(price, days, inv_level):
            if price <= 0:
                base = 200.0
            else:
                base = price

            if "High" in inv_level:
                inv_factor = 0.9
            elif "Medium" in inv_level:
                inv_factor = 1.0
            else:
                inv_factor = 1.1

            if days > 60:
                time_factor = 0.95
            elif days > 14:
                time_factor = 1.0
            else:
                time_factor = 1.05

            return round(base * inv_factor * time_factor, 2)

        # Helper to build model input row
        def build_model_input(days_until, inventory_label, current_price):
            inv_map = {
                "High (>2k)": 2,
                "Medium (500–2k)": 1,
                "Low (<500)": 0,
            }
            inv_code = inv_map.get(inventory_label, 1)

            # NOTE: Adjust these feature names to match your actual RF training features.
            # This is a simple example using three inputs.
            row = {
                "days_until_event": days_until,
                "inventory_level_code": inv_code,
                "current_price_input": current_price,
            }
            return pd.DataFrame([row])

        if run:
            # Use RF model if loaded; otherwise, fallback
            if rf_model is not None:
                input_df = build_model_input(days_until, inventory, current_price)
                try:
                    fair_value = float(rf_model.predict(input_df)[0])
                except Exception as e:
                    st.warning(
                        "Random Forest model loaded but prediction failed; "
                        "falling back to demo logic."
                    )
                    fair_value = demo_fair_value(current_price, days_until, inventory)
            else:
                fair_value = demo_fair_value(current_price, days_until, inventory)

            diff = current_price - fair_value

            # Predicted Fair Value card
            st.markdown(
                f"""
                <div style="
                    padding: 1rem 1.2rem;
                    border-radius: 0.75rem;
                    background: #111827;
                    border: 1px solid #4b5563;
                    margin-bottom: 0.75rem;
                ">
                    <div style="font-size: 0.9rem; color:#9ca3af;">Predicted Fair Value</div>
                    <div style="font-size: 2rem; font-weight: 700; color:#f9fafb;">
                        ${fair_value:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Recommendation bar
            if diff > 20:
                rec_label = "WAIT - Prices expected to drop"
                bg_color = "#78350f"
                border_color = "#fbbf24"
                text_color = "#fef9c3"
            elif -20 <= diff <= 20:
                rec_label = "FAIR - Listing is close to model value"
                bg_color = "#14532d"
                border_color = "#4ade80"
                text_color = "#bbf7d0"
            else:
                rec_label = "BUY - Current listing looks underpriced"
                bg_color = "#064e3b"
                border_color = "#34d399"
                text_color = "#a7f3d0"

            st.markdown(
                f"""
                <div style="
                    padding: 0.9rem 1.1rem;
                    border-radius: 0.75rem;
                    background: {bg_color};
                    border: 1px solid {border_color};
                    font-weight: 600;
                    margin-bottom: 1rem;
                    color: {text_color};
                ">
                    {rec_label}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Price Volatility Analysis
            st.markdown("### Price Volatility Analysis")
            st.caption("Predicted Historical Avg")

            days_back = np.arange(-30, 1)
            center = fair_value * 0.9
            volatility = center + np.sin(days_back / 4) * 10
            vol_df = pd.DataFrame(
                {"Days from Today": days_back, "Estimated Market Price": volatility}
            ).set_index("Days from Today")

            st.line_chart(vol_df)

            # Model comparison text (updated to Random Forest vs KNN)
            st.caption(
                "Model Comparison: Baseline k-NN vs Random Forest. "
                "Random Forest achieved the best RMSE and R² on held-out test data."
            )

            # Model Explainability (dummy feature importance)
            st.markdown("### Model Explainability (Feature Importance Example)")
            st.write("Which features drive the price prediction most? (Example)")

            feat_names = [
                "days_until_event",
                "inventory_level_code",
                "current_price_input",
            ]
            importance_values = [0.40, 0.30, 0.30]
            expl_df = (
                pd.DataFrame({"Feature": feat_names, "Importance": importance_values})
                .set_index("Feature")
            )

            st.bar_chart(expl_df)

        else:
            st.info("Set inputs on the left and click **Run Prediction** to see results.")


# ============= METHODOLOGY =============
with tab_methodology:
    st.header("Project Methodology")

    st.write(
        "To handle volatile ticket markets, we move beyond simple time-series forecasting "
        "and treat each market-year observation as a feature-rich data point influenced by "
        "market strength, sales momentum, and pricing dynamics."
    )

    st.markdown("#### 1. Data Acquisition")
    st.write(
        "We combine secondary-market pricing data with market-level features such as DMA rank, "
        "annual sales, and price premiums relative to average ticket prices."
    )

    st.markdown("#### 2. Model Selection")
    st.write(
        "- **Primary (Feature-Based):** Random Forest Regressor to capture nonlinear feature interactions.\n"
        "- **Baseline (Similarity-Based):** k-Nearest Neighbors as a benchmark that compares each market "
        "to similar historical markets."
    )

    st.markdown("#### 3. Future Work")
    st.write(
        "Future versions could incorporate sequence models or more granular intraday price data to track "
        "fine-grained volatility and dynamic pricing strategies."
    )

    st.markdown("### Performance Metrics (Example)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("R-Squared (Test)", "0.76")
    with col_m2:
        st.metric("RMSE (USD, Test)", "8.50")

    st.markdown("### Data Features Used (Example)")
    st.code(
        "year\n"
        "dma_rank\n"
        "annual_sales_current\n"
        "gross_growth\n"
        "ticket_growth\n"
        "sales_momentum\n"
        "avg_ticket_price_dma\n"
        "price_premium_vs_dma",
        language="text",
    )

    st.markdown("---")
    st.caption(
        "Fundamentals of Data Science • Fall 2025 Project\n"
        "Prototype for academic demonstration only."
    )


# ============= MARKET AGGREGATOR =============
with tab_aggregator:
    st.header("Live Market Aggregator")
    st.write("Real-time lowest prices across platforms (example data).")

    if "platform_data" not in st.session_state:
        st.session_state.platform_data = pd.DataFrame(
            {
                "Platform": ["StubHub", "SeatGeek", "TickPick", "VividSeats"],
                "Section/Row": [
                    "Floor A / Row 5",
                    "Lower 112 / Row 10",
                    "Club 210 / Row 3",
                    "Upper 332 / Row 15",
                ],
                "Price (w/ Fees)": [265.0, 240.0, 255.0, 220.0],
                "Action": ["Best Value", "", "", "Best Value"],
            }
        )

    if st.button("Refresh Data"):
        df = st.session_state.platform_data.copy()
        noise = np.random.uniform(-10, 10, size=len(df))
        df["Price (w/ Fees)"] = (df["Price (w/ Fees)"] + noise).round(2)
        df["Price (w/ Fees)"] = df["Price (w/ Fees)"].clip(lower=0)
        st.session_state.platform_data = df

    st.dataframe(
        st.session_state.platform_data,
        use_container_width=True,
    )

    st.caption(
        'Note: Prices include estimated service fees. '
        '"Best Value" is based on section quality relative to price (example logic).'
    )
