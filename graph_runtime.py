from simulation_utils import calculate_infection_probability, determine_recovery_or_death
import random
import numpy as np
import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance (in km) between two lat/lon points"""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_daily_graph(data, adjacency_list, infection_log=None):
    """
    Use precomputed adjacency list to create infection/contact graph
    Color-code edges based on infection/recovery status
    """
    edges = adjacency_list
    edge_list = []
    data_dict = {p["id"]: p for p in data}
    sampling_rate = 1

    for p1_id, neighbors in adjacency_list.items():
        if random.random() < sampling_rate:
            p1 = data_dict.get(int(p1_id), None)
            for p2_id in neighbors[:3]:  # limit neighbors for visual clarity
                p2 = data_dict[int(p2_id)]
                status1 = p1["status"]
                status2 = p2["status"]

                if status1 == "infected" and status2 == "infected":
                    color = [255, 182, 193]  # Dark blue
                elif status1 == "recovered" and status2 == "recovered":
                    color = [230, 230, 250]  # Earthy orange
                elif {status1, status2}.issubset({"Healthy", "recovered"}):
                    color = [180, 180, 180]  # Light gray
                else:
                    continue  # Skip noisy edge cases

                edge_list.append({
                    "source": [p1["lon"], p1["lat"]],
                    "target": [p2["lon"], p2["lat"]],
                    "color": color
                })

    if infection_log:
        for src, tgt in infection_log:
            if src in data_dict and tgt in data_dict:
                edge_list.append({
                    "source": [data_dict[src]["lon"], data_dict[src]["lat"]],
                    "target": [data_dict[tgt]["lon"], data_dict[tgt]["lat"]],
                    "color": [48, 25, 52]  # Dark purple for logged infections
                })

    return edges, edge_list
import random
import math

def euclidean_distance(person1, person2):
    """ Calculate Euclidean distance between two people based on lat/lon. """
    return math.sqrt((person1["lat"] - person2["lat"]) ** 2 + (person1["lon"] - person2["lon"]) ** 2)

def infect_random_start(data, per_cluster=3, num_clusters=5):
    """ Infect people in 5 randomly selected clusters within the Punjab bounds. """
    punjab_lat_min, punjab_lat_max = 27.0, 34.0
    punjab_lon_min, punjab_lon_max = 68.0, 75.0

    # Filter only people within Punjab
    punjab_people = [p for p in data if punjab_lat_min <= p["lat"] <= punjab_lat_max and punjab_lon_min <= p["lon"] <= punjab_lon_max]

    if len(punjab_people) < per_cluster:
        per_cluster = len(punjab_people)

    # Choose num_clusters random people as the center of each cluster
    random_centers = random.sample(punjab_people, num_clusters)

    # Infect the center person and the nearest people in the cluster
    for center_person in random_centers:
        # Sort people by distance to the center
        sorted_by_distance = sorted(punjab_people, key=lambda p: euclidean_distance(center_person, p))
        
        # Infect the center person and the nearest per_cluster people
        infected = sorted_by_distance[:per_cluster]
        for person in infected:
            person["status"] = "infected"
            person["days_infected"] = 0

    return data

def run_simulation_step(data, G, local_density=None, base_prob=0.05, infection_log=None):
    if infection_log is None:
        infection_log = []

    data_dict = {p["id"]: p for p in data}
    
    # Remove nodes without neighbors
    nodes_to_remove = []
    for person in data:
        # Get neighbors from the graph (G) for this person
        neighbors = G.get(str(person["id"]), [])
        if not neighbors:  # If no neighbors, mark for removal
            nodes_to_remove.append(person["id"])
    
    # Remove the nodes without neighbors from data
    data = [person for person in data if person["id"] not in nodes_to_remove]
    data_dict = {p["id"]: p for p in data}  # Rebuild the dictionary after removal

    # Update the graph to remove edges connected to removed nodes
    G = {key: [nid for nid in neighbors if nid not in nodes_to_remove] 
         for key, neighbors in G.items() if key not in nodes_to_remove}

    # Step 1: Handle infected people and progression of infection (recovery or death)
    for person in data:
        if person["status"] == "infected":
            person["days_infected"] += 1
            person["status"] = determine_recovery_or_death(person, person["days_infected"])

    # Step 2: Attempt to infect healthy people based on their infected neighbors
    for person in data:
        if person["status"] != "Healthy":
            continue

        neighbors = G.get(str(person["id"]), [])
        infected_contacts = [nid for nid in neighbors if data_dict[int(nid)]["status"] == "infected"]

        if infected_contacts:
            local_pop = local_density.get(person["id"], 0) if local_density else len(neighbors)
            f = len(infected_contacts)
            prob = calculate_infection_probability(len(infected_contacts), f, person, local_pop, base_prob)

            if random.random() < prob:
                person["status"] = "infected"
                person["days_infected"] = 0
                source = int(random.choice(infected_contacts))
                infection_log.append((source, person["id"]))

    return data, G, infection_log
