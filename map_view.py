import streamlit as st
import pandas as pd
import pydeck as pdk
import json
import os
import subprocess
from collections import Counter
from graph_runtime import build_daily_graph, run_simulation_step, infect_random_start

st.set_page_config(layout="wide", page_title="Pakistan COVID Simulation")
st.title("🧬 Pakistan Infectious Disease Spread Simulation")

# State initialization
if "data_created" not in st.session_state:
    st.session_state.data_created = os.path.exists("synthetic_population.json") and os.path.exists("adjacency_list.json")
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "infection_log" not in st.session_state:
    st.session_state.infection_log = []
if "selected_day" not in st.session_state:
    st.session_state.selected_day = 0

# Only button: Run Simulation
if st.sidebar.button("▶️ Run Simulation") and st.session_state.data_created:
    st.session_state.simulation_done = False
    st.session_state.timeline = []
    st.session_state.infection_log = []

    with open("synthetic_population.json") as f:
        base_data = json.load(f)
    with open("adjacency_list.json") as f:
        adjacency = json.load(f)

    data = infect_random_start(base_data, per_cluster=5, num_clusters=10)
    for day in range(31):
        G, edge_list = build_daily_graph(data, adjacency, st.session_state.infection_log)
        data, G, st.session_state.infection_log = run_simulation_step(data, G, None, 0.05, st.session_state.infection_log)

        st.session_state.timeline.append({
            "data": json.loads(json.dumps(data)),
            "edges": edge_list
        })
    st.session_state.simulation_done = True
    st.success("✅ Simulation complete.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📍 Simulation", "📈 Statistics", "📊 Impact Analysis"])

with tab1:
    if st.session_state.simulation_done:
        st.session_state.selected_day = st.sidebar.slider("📅 Select Day", 0, 30, st.session_state.selected_day)
    else:
        st.sidebar.info("ℹ️ Run the simulation to enable day slider.")

    # Load snapshot
    map_data = None
    edge_data = pd.DataFrame(columns=["source", "target", "color"])

    if st.session_state.simulation_done:
        snapshot = st.session_state.timeline[st.session_state.selected_day]
        map_data = pd.DataFrame(snapshot["data"])
        edge_data = pd.DataFrame(snapshot["edges"])
    elif st.session_state.data_created:
        with open("synthetic_population.json") as f:
            map_data = pd.DataFrame(json.load(f))

    # Super-spreaders
    if map_data is not None and not map_data.empty:
        if st.session_state.simulation_done:
            counts = Counter(i for i, _ in st.session_state.infection_log)
            super_spreaders = {pid for pid, c in counts.items() if c >= 5}
            map_data["super_spreader"] = map_data["id"].apply(lambda x: x in super_spreaders)
        else:
            map_data["super_spreader"] = False

        def get_color(row):
            if row["super_spreader"]:
                return [255, 140, 0]  # 🟠
            return {
                "Healthy": [0, 255, 0],
                "infected": [255, 0, 0],
                "recovered": [0, 168, 150],
                "deceased": [255, 255, 255]
            }.get(row["status"], [128, 128, 128])

        map_data["color"] = map_data.apply(get_color, axis=1)
        map_data["size"] = map_data["status"].map({
            "infected": 300,
            "Healthy": 200,
            "recovered": 250,
            "deceased": 150
        }).fillna(100)
    else:
        map_data = pd.DataFrame(columns=["id", "lat", "lon", "status", "super_spreader", "color", "size"])

    view = pdk.ViewState(latitude=30.3753, longitude=69.3451, zoom=5.2)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position='[lon, lat]',
            get_color='color',
            get_radius='size',
            pickable=True
        )
    ]

    if st.session_state.simulation_done and not edge_data.empty:
        layers.append(
            pdk.Layer(
                "LineLayer",
                edge_data,
                get_source_position="source",
                get_target_position="target",
                get_color="color",
                get_width=2
            )
        )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "ID: {id}\nStatus: {status}"}
    ))

    # Dashboard
    st.subheader(f"📊 Day {st.session_state.selected_day} Statistics")
    if st.session_state.simulation_done:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Healthy", int((map_data["status"] == "Healthy").sum()))
        col2.metric("🔴 Infected", int((map_data["status"] == "infected").sum()))
        col3.metric("🔵 Recovered", int((map_data["status"] == "recovered").sum()))
        col4.metric("⚪ Deceased", int((map_data["status"] == "deceased").sum()))
    else:
        st.info("ℹ️ Run simulation to populate dashboard.")

with tab2:
    st.header("📈 Timeline of Statuses (All Days)")
    if st.session_state.simulation_done:
        daywise_stats = {
            "Day": [],
            "Healthy": [],
            "Infected": [],
            "Recovered": [],
            "Deceased": []
        }

        for i, snapshot in enumerate(st.session_state.timeline):
            df = pd.DataFrame(snapshot["data"])
            daywise_stats["Day"].append(i)
            daywise_stats["Healthy"].append((df["status"] == "Healthy").sum())
            daywise_stats["Infected"].append((df["status"] == "Infected").sum())
            daywise_stats["Recovered"].append((df["status"] == "Recovered").sum())
            daywise_stats["Deceased"].append((df["status"] == "Deceased").sum())

        stats_df = pd.DataFrame(daywise_stats)
        st.line_chart(stats_df.set_index("Day"))

        csv = stats_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="daily_stats.csv", mime="text/csv")
    else:
        st.info("ℹ️ Run simulation to view full statistics.")

with tab3:
    st.header("📊 Impact Analysis by Age Group")
    if st.session_state.simulation_done:
        latest_df = pd.DataFrame(st.session_state.timeline[-1]["data"])

        st.subheader("🟦 Status Distribution by Age Group")
        age_stats = latest_df.groupby(["age_group", "status"]).size().unstack(fill_value=0)
        st.bar_chart(age_stats)

        csv1 = age_stats.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Bar Chart Data", csv1, "age_status_distribution.csv", "text/csv")

        st.subheader("📈 Infected Trend over Time by Age Group")
        infected_over_time = {
            "Children": [],
            "Adults": [],
            "Elderly": []
        }

        for day_snapshot in st.session_state.timeline:
            df = pd.DataFrame(day_snapshot["data"])
            day_counts = df[df["status"] == "Infected"].groupby("age_group").size()
            for group in infected_over_time:
                infected_over_time[group].append(day_counts.get(group, 0))

        line_data = pd.DataFrame(infected_over_time)
        line_data.index.name = "Day"
        st.line_chart(line_data)

        csv2 = line_data.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Line Chart Data", csv2, "infected_trend.csv", "text/csv")

        st.subheader("🥧 Final Day Infected Distribution by Age Group")
        infected_pie = latest_df[latest_df["status"] == "Infected"]["age_group"].value_counts()
        fig = infected_pie.plot.pie(autopct="%1.1f%%", title="Infected by Age Group").figure
        st.pyplot(fig)

        pie_csv = infected_pie.reset_index()
        pie_csv.columns = ["Age Group", "Infected Count"]
        pie_csv = pie_csv.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Pie Chart Data", pie_csv, "infected_pie_chart.csv", "text/csv")

        st.caption("This section shows infection dynamics across Children, Adults, and Elderly age groups.")
    else:
        st.info("ℹ️ Run simulation to analyze age group impact.")