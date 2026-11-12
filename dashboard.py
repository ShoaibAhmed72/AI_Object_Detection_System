"""
Streamlit dashboard: live status + historical statistics + search.

Run with:   streamlit run dashboard.py

This process is independent from main.py -- it only reads from MongoDB,
so you can leave main.py running the camera in one terminal and this
dashboard running in another (or on another machine, if MONGO_URI points
at a shared/remote MongoDB).
"""

import datetime

import pandas as pd
import streamlit as st

import config
from database import Database




@st.cache_resource
def get_db():
    return Database()


db = get_db()

@st.cache_data(ttl=5)
def cached_get_live_state():
    return db.get_live_state()

@st.cache_data(ttl=5)
def cached_get_statistics():
    return db.get_statistics()

@st.cache_data(ttl=5)
def cached_search_detections(limit=500):
    return db.search_detections(limit=limit)

@st.cache_data(ttl=5)
def cached_search_scenes(limit=500):
    return db.search_scenes(limit=limit)

@st.cache_data(ttl=5)
def cached_get_relationship_totals():
    return db.get_relationship_totals()
# --------------------------------------------------------------------------
# Page Configuration & Modern Styling
# --------------------------------------------------------------------------
st.set_page_config(page_title=config.PROJECT_NAME, page_icon="📷", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            text-align: center;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.4rem 1rem;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return Database()


db = get_db()

st.title("📷 " + config.PROJECT_NAME)

if not db.connected:
    st.error(
        "Not connected to MongoDB. Check that MongoDB is running and that "
        f"MONGO_URI is correct (currently: {config.MONGO_URI})."
    )

tab_live, tab_history, tab_scenes, tab_relationships, tab_search, tab_admin = st.tabs(
    ["🔴 Live", "📋 Detection History", "🏠 Scene History", "🔗 Relationships", "🔍 Search", "⚙️ Admin"]
)

# --------------------------------------------------------------------------
with tab_live:
    st.subheader("Current Status")
    # live = db.get_live_state()

    live = cached_get_live_state()
    stats = cached_get_statistics()
    if live:
        col1, col2, col3 = st.columns(3)
        col1.metric("Scene", live.get("scene") or "—",
                    f'{(live.get("scene_confidence") or 0) * 100:.0f}% confidence')
        col2.metric("Objects Visible", len(live.get("objects", [])))
        col3.metric("FPS", live.get("fps", 0))

        st.write("**Currently Detected Objects**")
        st.write(", ".join(live.get("objects", [])) or "None")

        st.write("**Objects Together**")
        together = live.get("objects_together", [])
        st.write(" + ".join(together) if together else "—")

        st.caption(f'Last updated: {live.get("updated_at")}')
    else:
        st.info("No live data yet. Start `python main.py` to begin streaming detections.")

    st.subheader("Detection Statistics")
    stats = db.get_statistics()
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Detection Sessions", stats.get("total_detection_sessions", 0))
        c2.metric("Unique Objects", stats.get("unique_objects", 0))
        c3.metric("Most Detected Object", stats.get("most_detected_object") or "—")
        c4.metric("Longest Detection (s)", stats.get("longest_detection_seconds", 0))

        c5, c6, c7 = st.columns(3)
        c5.metric("Total Scene Sessions", stats.get("total_scene_sessions", 0))
        c6.metric("Most Common Scene", stats.get("most_common_scene") or "—")
        c7.metric("Longest Scene", f'{stats.get("longest_scene_name") or "—"} '
                                   f'({stats.get("longest_scene_seconds", 0)}s)')
    else:
        st.info("No statistics yet.")

# --------------------------------------------------------------------------
with tab_history:
    st.subheader("Detection History")
    # docs = db.search_detections(limit=500)
    docs = cached_search_detections(limit=500)
    if docs:
        from pdf_exporter import generate_objects_pdf
        
        if st.button("📄 Generate & Download Objects PDF Report"):
            pdf_path = generate_objects_pdf(db)
            if pdf_path:
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Click here to download PDF",
                        data=pdf_file,
                        file_name="detected_objects_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No records available to export.")
                
        st.markdown("<br>", unsafe_allow_html=True)
        df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
        st.dataframe(df, width='stretch')
    else:
        st.info("No detection history yet.")

# --------------------------------------------------------------------------
with tab_scenes:
    st.subheader("Scene History")
    # docs = db.search_scenes(limit=500)
    docs = cached_search_scenes(limit=500)
    if docs:
        df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
        st.dataframe(df, width='stretch')
    else:
        st.info("No scene history yet.")

# --------------------------------------------------------------------------
with tab_relationships:
    st.subheader("Total Time Together (all-time)")
    # totals = db.get_relationship_totals()
    totals = cached_get_relationship_totals()
    if totals:
        rows = [
            {"Objects": " + ".join(t["_id"]), "Total Time (s)": round(t["total_seconds"], 1),
             "Sessions": t["sessions"]}
            for t in totals
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch')
    else:
        st.info("No relationship data yet.")

    st.subheader("Recent Relationship Sessions")
    docs = db.search_relationships(limit=200)
    if docs:
        df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
        df["Objects"] = df["objects"].apply(lambda o: " + ".join(o))
        st.dataframe(df.drop(columns=["objects"]), width='stretch')

# --------------------------------------------------------------------------
with tab_search:
    st.subheader("Search Detection History")
    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        object_name = col1.text_input("Object name")
        tracking_id = col2.text_input("Tracking ID")
        date = col3.date_input("Date", value=None)

        col4, col5 = st.columns(2)
        min_duration = col4.number_input("Minimum duration (s)", min_value=0.0, value=0.0)
        min_confidence = col5.slider("Minimum confidence", 0.0, 1.0, 0.0)

        submitted = st.form_submit_button("Search")

    if submitted:
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime.date) else None
        results = db.search_detections(
            object_name=object_name or None,
            tracking_id=tracking_id or None,
            date=date_str,
            min_duration=min_duration or None,
            min_confidence=min_confidence or None,
        )
        st.write(f"{len(results)} result(s)")
        if results:
            df = pd.DataFrame(results).drop(columns=["_id"], errors="ignore")
            st.dataframe(df, width='stretch')

# --------------------------------------------------------------------------
with tab_admin:
    st.subheader("Admin")
    st.warning("This permanently deletes all detection, scene, and relationship history.")
    if st.button("Clear All History"):
        if db.clear_history():
            st.success("History cleared.")
        else:
            st.error("Failed to clear history (DB offline?).")