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
    sampling_rate = 0.01

    for p1_id, neighbors in adjacency_list.items():
        if random.random() < sampling_rate:
            p1 = data_dict[int(p1_id)]
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

def infect_random_start(data, per_city=2):
    """
    Infect `per_city` people in each of the 10 major cities of Pakistan.
    """

    city_bounds = [
        ("Karachi",     24.8, 25.1, 66.9, 67.3),
        ("Lahore",      31.4, 31.7, 74.2, 74.4),
        ("Faisalabad",  31.3, 31.5, 72.9, 73.2),
        ("Rawalpindi",  33.5, 33.7, 73.0, 73.2),
        ("Multan",      30.1, 30.3, 71.4, 71.6),
        ("Hyderabad",   25.3, 25.5, 68.3, 68.5),
        ("Islamabad",   33.6, 33.8, 72.9, 73.1),
        ("Gujranwala",  32.1, 32.2, 74.1, 74.2),
        ("Peshawar",    34.0, 34.2, 71.4, 71.6),
        ("Quetta",      30.1, 30.3, 66.9, 67.1)
    ]

    infected_ids = []

    for city_name, lat_min, lat_max, lon_min, lon_max in city_bounds:
        candidates = [
            p["id"] for p in data
            if lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max and p["id"] not in infected_ids
        ]
        selected = random.sample(candidates, min(per_city, len(candidates)))
        infected_ids.extend(selected)

    for p in data:
        if p["id"] in infected_ids:
            p["status"] = "infected"
            p["days_infected"] = 0
            p["initial_cluster_infected"] = True

    return data

def run_simulation_step(data, G, local_density=None, base_prob=0.05, infection_log=None):
    if infection_log is None:
        infection_log = []

    data_dict = {p["id"]: p for p in data}

    for person in data:
        if person["status"] == "infected":
            person["days_infected"] += 1
            person["status"] = determine_recovery_or_death(person, person["days_infected"])

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

    return data, infection_log
