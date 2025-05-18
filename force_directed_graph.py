import streamlit as st
import pandas as pd
import pydeck as pdk
import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from graph_runtime import build_daily_graph, run_simulation_step, infect_random_start

st.set_page_config(layout="wide", page_title="Pakistan COVID Hybrid Simulation")
st.title("🧬 Pakistan COVID-19 Hybrid Geo-Network Simulation")

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
if "graph_layouts" not in st.session_state:
    st.session_state.graph_layouts = {}
if "network_metrics" not in st.session_state:
    st.session_state.network_metrics = {}
if "force_layout_iterations" not in st.session_state:
    st.session_state.force_layout_iterations = 5

# Functions for hybrid geo-network approach
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two lat/lon points"""
    R = 6371  # Earth radius in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_hybrid_network(data, adjacency_list):
    """Create a NetworkX graph from data maintaining geographic positions"""
    # Create a dict for faster lookup
    data_dict = {str(p["id"]): p for p in data}
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with geographic positions
    for person in data:
        G.add_node(
            str(person["id"]),
            pos=(person["lon"], person["lat"]),  # Geographic position
            status=person["status"],
            age_group=person["age_group"],
            days_infected=person["days_infected"] if person["status"] == "infected" else 0
        )
    
    # Add edges from adjacency list
    edge_count = 0
    for p1_id, neighbors in adjacency_list.items():
        if edge_count > 50000:  # Limit edges for visual clarity
            break
            
        if p1_id in G:
            # Only add a subset of connections for visual clarity
            sampled_neighbors = neighbors[:min(5, len(neighbors))]
            for p2_id in sampled_neighbors:
                if str(p2_id) in G:
                    # Add edge weight based on infection status
                    p1 = data_dict.get(p1_id)
                    p2 = data_dict.get(str(p2_id))
                    
                    if p1 and p2:
                        # Higher weight for connections between infected people
                        if p1["status"] == "infected" and p2["status"] == "infected":
                            weight = 3.0
                        elif p1["status"] == "infected" or p2["status"] == "infected":
                            weight = 2.0
                        else:
                            weight = 1.0
                            
                        G.add_edge(p1_id, str(p2_id), weight=weight)
                        edge_count += 1
    
    return G

def generate_hybrid_layout(G, day, influence=0.3):
    """Generate layout that combines geographic positions with force-directed adjustments"""
    # Start with geographic positions
    geo_pos = nx.get_node_attributes(G, 'pos')
    
    # Check if we already have this layout cached 
    if day in st.session_state.graph_layouts:
        return st.session_state.graph_layouts[day]
    
    # Apply force-directed algorithm with geographic positions as starting point
    pos = nx.spring_layout(
        G, 
        pos=geo_pos,  # Start from geographic positions  
        iterations=st.session_state.force_layout_iterations,  # Fewer iterations to keep geographic influence
        k=influence,  # Controls the distance between nodes
        seed=42  # For reproducibility
    )
    
    # Cache this layout
    st.session_state.graph_layouts[day] = pos
    return pos

def prepare_network_deck_data(G, pos, data):
    """Prepare data for Pydeck visualization of the network"""
    # Create nodes dataframe
    nodes_data = []
    for node_id in G.nodes():
        node_pos = pos[node_id]
        person = next((p for p in data if str(p["id"]) == node_id), None)
        
        if person:
            # Determine color and size based on status
            if person["status"] == "infected":
                color = [255, 0, 0, 200]  # Red
                size = 100  # Reduced size for better visibility
            elif person["status"] == "recovered":
                color = [0, 255, 0, 200]  # Green
                size = 80  
            elif person["status"] == "deceased":
                color = [255, 255, 255, 200]  # White
                size = 60
            else:  # Healthy
                color = [0, 0, 255, 200]  # Blue
                size = 80
            
            # Use geographic coordinates for visualization
            # Swap lon/lat for position as PyDeck expects [lon, lat]
            position = [float(person["lon"]), float(person["lat"])]
            
            nodes_data.append({
                "id": node_id,
                "position": position,
                "color": color,
                "size": size,
                "status": person["status"],
                "age_group": person["age_group"]
            })
    
    # Create edges dataframe - use actual geographic coordinates
    edges_data = []
    for source, target in G.edges():
        source_person = next((p for p in data if str(p["id"]) == source), None)
        target_person = next((p for p in data if str(p["id"]) == target), None)
        
        if source_person and target_person:
            # Determine edge color based on infection status
            if source_person["status"] == "infected" and target_person["status"] == "infected":
                color = [255, 0, 0, 150]  # Red
                width = 5
            elif source_person["status"] == "infected" or target_person["status"] == "infected":
                color = [255, 165, 0, 150]  # Orange
                width = 4
            else:
                color = [100, 100, 100, 100]  # Gray, increased opacity
                width = 2
            
            # Use the actual geographic coordinates
            source_position = [float(source_person["lon"]), float(source_person["lat"])]
            target_position = [float(target_person["lon"]), float(target_person["lat"])]
            
            edges_data.append({
                "source": source_position,
                "target": target_position,
                "color": color,
                "width": width
            })
    
    return pd.DataFrame(nodes_data), pd.DataFrame(edges_data)

def calculate_network_metrics(G, data):
    """Calculate network metrics for the current graph"""
    metrics = {}
    
    # Basic metrics
    metrics["nodes"] = len(G.nodes())
    metrics["edges"] = len(G.edges())
    metrics["avg_degree"] = 2 * metrics["edges"] / metrics["nodes"] if metrics["nodes"] > 0 else 0
    
    # Calculate average clustering coefficient (local transitivity)
    try:
        metrics["clustering"] = nx.average_clustering(G)
    except:
        metrics["clustering"] = 0
    
    # Count status distribution
    status_counts = Counter(person["status"] for person in data)
    metrics["healthy"] = status_counts.get("Healthy", 0)
    metrics["infected"] = status_counts.get("infected", 0)
    metrics["recovered"] = status_counts.get("recovered", 0)
    metrics["deceased"] = status_counts.get("deceased", 0)
    
    # Connected components
    try:
        metrics["components"] = nx.number_connected_components(G)
    except:
        metrics["components"] = "N/A"
    
    return metrics

# Simulation Button
st.sidebar.header("Simulation Controls")
force_influence = st.sidebar.slider(
    "Force Layout Strength", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.3,
    help="Higher values make the network more force-directed, lower values keep it closer to geographic positions"
)

force_iterations = st.sidebar.slider(
    "Force Layout Iterations",
    min_value=1,
    max_value=20,
    value=st.session_state.force_layout_iterations,
    help="More iterations result in a more stable force-directed layout"
)
st.session_state.force_layout_iterations = force_iterations

if st.sidebar.button("▶️ Run Simulation") and st.session_state.data_created:
    st.session_state.simulation_done = False
    st.session_state.timeline = []
    st.session_state.infection_log = []
    st.session_state.graph_layouts = {}
    st.session_state.network_metrics = {}
    
    with st.spinner("Running simulation..."):
        with open("synthetic_population.json") as f:
            base_data = json.load(f)
        with open("adjacency_list.json") as f:
            adjacency = json.load(f)

        data = infect_random_start(base_data, per_cluster=5, num_clusters=10)
        
        # First day's data
        st.session_state.timeline.append(json.loads(json.dumps(data)))
        
        # Run simulation for 31 days
        for day in range(1, 31):
            # Create the graph for current state
            G = adjacency
            data, G, st.session_state.infection_log = run_simulation_step(data, G, None, 0.05, st.session_state.infection_log)
            
            # Store this day's data
            st.session_state.timeline.append(json.loads(json.dumps(data)))
            
            # Calculate and store network metrics
            network_G = create_hybrid_network(data[:50000], {k: adjacency[k] for k in list(adjacency)[:50000]})
            st.session_state.network_metrics[day] = calculate_network_metrics(network_G, data)
    
    st.session_state.simulation_done = True
    st.success("✅ Simulation complete.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🌐 Hybrid Geo-Network", "📈 Statistics", "📊 Impact Analysis"])

with tab1:
    if st.session_state.simulation_done:
        st.session_state.selected_day = st.sidebar.slider("📅 Select Day", 0, 30, st.session_state.selected_day)
        
        # Get the current day's data
        data = st.session_state.timeline[st.session_state.selected_day]
        
        # Load adjacency list
        with open("adjacency_list.json") as f:
            adjacency = json.load(f)
        
        # Create network with sampling for performance
        with st.spinner("Generating network visualization..."):
            sample_size = min(50000, len(data))
            sampled_data = data[:sample_size]
            sampled_adjacency = {k: adjacency[k] for k in list(adjacency)[:sample_size] if k in adjacency}
            
            G = create_hybrid_network(sampled_data, sampled_adjacency)
            pos = generate_hybrid_layout(G, st.session_state.selected_day, force_influence)
            nodes_df, edges_df = prepare_network_deck_data(G, pos, sampled_data)
        
        # Initial view based on data points
        center_lat = np.mean([p["lat"] for p in sampled_data])
        center_lon = np.mean([p["lon"] for p in sampled_data])
        
        # Create visualization with Pydeck
        view_state = pdk.ViewState(
            latitude=30.3753,  # Center of Pakistan
            longitude=69.3451,
            zoom=5,
            pitch=45,
            bearing=0
        )
        
        # Create the deck with improved visualization settings
        deck = pdk.Deck(
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=nodes_df,
                    get_position="position",
                    get_color="color",
                    get_radius="size",
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    radius_scale=6,
                    radius_min_pixels=3,
                    radius_max_pixels=30,
                    line_width_min_pixels=1,
                ),
                pdk.Layer(
                    "LineLayer",
                    data=edges_df,
                    get_source_position="source",
                    get_target_position="target",
                    get_color="color",
                    get_width="width",
                    width_scale=2,
                    width_min_pixels=2,
                    pickable=True,
                ),
            ],
            initial_view_state=view_state,
            tooltip={"text": "{id} ({status})"},
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        
        # Display the deck
        st.pydeck_chart(deck)
        
        # Network metrics
        st.subheader("📊 Network Analysis")
        
        # Dynamic metrics from this network
        metrics = calculate_network_metrics(G, sampled_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", metrics["nodes"])
        with col2:
            st.metric("Edges", metrics["edges"])
        with col3:
            st.metric("Avg. Connections", round(metrics["avg_degree"], 2))
        with col4:
            st.metric("Clustering Coefficient", round(metrics["clustering"], 4))
        
        # Dashboard
        st.subheader(f"📊 Day {st.session_state.selected_day} Disease Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Healthy", metrics["healthy"])
        col2.metric("🔴 Infected", metrics["infected"])
        col3.metric("🔵 Recovered", metrics["recovered"])
        col4.metric("⚪ Deceased", metrics["deceased"])
    else:
        st.info("ℹ️ Run simulation to view hybrid geo-network.")

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

        for i, day_data in enumerate(st.session_state.timeline):
            df = pd.DataFrame(day_data)
            daywise_stats["Day"].append(i)
            daywise_stats["Healthy"].append((df["status"] == "Healthy").sum())
            daywise_stats["Infected"].append((df["status"] == "infected").sum())
            daywise_stats["Recovered"].append((df["status"] == "recovered").sum())
            daywise_stats["Deceased"].append((df["status"] == "deceased").sum())

        stats_df = pd.DataFrame(daywise_stats)
        st.line_chart(stats_df.set_index("Day"))

        csv = stats_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="daily_stats.csv", mime="text/csv")
        
        # Network metrics over time
        if st.session_state.network_metrics:
            st.subheader("🕸️ Network Evolution")
            
            network_stats = {
                "Day": list(st.session_state.network_metrics.keys()),
                "Avg Degree": [m["avg_degree"] for m in st.session_state.network_metrics.values()],
                "Clustering": [m["clustering"] for m in st.session_state.network_metrics.values()]
            }
            
            network_df = pd.DataFrame(network_stats)
            st.line_chart(network_df.set_index("Day"))
    else:
        st.info("ℹ️ Run simulation to view statistics.")

with tab3:
    st.header("📊 Impact Analysis")
    if st.session_state.simulation_done:
        latest_df = pd.DataFrame(st.session_state.timeline[-1])

        # Status distribution by age group
        st.subheader("🟦 Status Distribution by Age Group")
        age_stats = latest_df.groupby(["age_group", "status"]).size().unstack(fill_value=0)
        st.bar_chart(age_stats)

        csv1 = age_stats.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Age Group Data", csv1, "age_status_distribution.csv", "text/csv")

        # Super spreaders analysis
        st.subheader("🔥 Super Spreader Analysis")
        
        if st.session_state.infection_log:
            counts = Counter(i for i, _ in st.session_state.infection_log)
            top_spreaders = counts.most_common(10)
            
            spreader_df = pd.DataFrame(top_spreaders, columns=["Person ID", "Infections Caused"])
            
            # Get additional info about top spreaders
            for i, row in spreader_df.iterrows():
                person_id = row["Person ID"]
                person_data = next((p for p in latest_df.to_dict('records') if p["id"] == person_id), None)
                
                if person_data:
                    spreader_df.at[i, "Age Group"] = person_data["age_group"]
                    spreader_df.at[i, "Chronic Illness"] = "Yes" if person_data["chronic_illness"] else "No"
                    spreader_df.at[i, "Status"] = person_data["status"]
                    spreader_df.at[i, "Daily Contacts"] = person_data["daily_contacts"]
            
            st.table(spreader_df)
            
            # Plot top spreader locations on map
            st.subheader("🗺️ Super Spreader Locations")
            
            top_spreader_ids = [id for id, _ in top_spreaders]
            spreader_locations = []
            
            for person in latest_df.to_dict('records'):
                if person["id"] in top_spreader_ids:
                    spreader_locations.append({
                        "lat": person["lat"],
                        "lon": person["lon"],
                        "infections": counts[person["id"]],
                        "age_group": person["age_group"]
                    })
            
            if spreader_locations:
                spreader_df = pd.DataFrame(spreader_locations)
                
                view = pdk.ViewState(latitude=30.3753, longitude=69.3451, zoom=5.2)
                
                st.pydeck_chart(pdk.Deck(
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            spreader_df,
                            get_position=["lon", "lat"],
                            get_radius="infections * 1000",  # Size by number of infections
                            get_fill_color=[255, 140, 0, 200],  # Orange
                            pickable=True
                        )
                    ],
                    initial_view_state=view,
                    tooltip={"text": "Infections: {infections}\nAge Group: {age_group}"}
                ))
        else:
            st.info("No infection data available.")
    else:
        st.info("ℹ️ Run simulation to analyze impact.")

if not st.session_state.data_created:
    st.error("❌ Data files not found. Please run data_creation.py first.")