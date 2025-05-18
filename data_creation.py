import json
import random
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.spatial import KDTree
import time
import os
from tqdm import tqdm

# Constants
NUM_PEOPLE = 30000
RADIUS_KM = 5.0
BATCH_SIZE = 50000  # Still safe as it's > NUM_PEOPLE
TEMP_DIR = "temp_adjacency"
ADJ_INDEX_FILE = "adjacency_index.json"

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Helper functions
def load_punjab_boundary():
    print("⏳ Loading Punjab boundary from Pakistan geojson...")
    gdf = gpd.read_file("pakistan.geojson")
    
    # Ensure Punjab is selected — depends on the actual file's 'name' or 'province' column
    punjab = gdf[gdf['NAME_1'].str.contains("Punjab", case=False, na=False)]
    
    if punjab.empty:
        raise ValueError("Punjab boundary not found in the geojson. Check attribute names.")

    print("✅ Punjab boundary loaded")
    return punjab.unary_union

def generate_valid_location(punjab_polygon):
    bounds = punjab_polygon.bounds
    while True:
        lat = random.uniform(bounds[1], bounds[3])
        lon = random.uniform(bounds[0], bounds[2])
        if punjab_polygon.contains(Point(lon, lat)):
            return lat, lon

def assign_age_group():
    return random.choices(["Child", "Adult", "Elderly"], weights=[30, 45, 25])[0]

def assign_chronic_illness():
    return random.random() < 0.8

def assign_distancing():
    return random.random() < 0.2

def degree_to_km(lat):
    return 111.32 * np.cos(np.radians(lat))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def generate_population():
    punjab_polygon = load_punjab_boundary()
    individuals = []
    coords = []

    print(f"⏳ Generating {NUM_PEOPLE} individuals within Punjab...")
    for i in tqdm(range(NUM_PEOPLE)):
        lat, lon = generate_valid_location(punjab_polygon)
        age_group = assign_age_group()
        chronic = assign_chronic_illness()
        distancing = assign_distancing()

        # Early detection factor (0.0 to 1.0)
        if age_group == "Adult":
            early_detect = round(random.uniform(0.6, 0.9), 2)
        elif age_group == "Elderly":
            early_detect = round(random.uniform(0.4, 0.7), 2)
        else:  # Child
            early_detect = round(random.uniform(0.3, 0.6), 2)

        if chronic:
            early_detect = max(0.0, early_detect - 0.1)

        person = {
            "id": i,
            "lat": lat,
            "lon": lon,
            "age_group": age_group,
            "chronic_illness": chronic,
            "distancing": distancing,
            "status": "Healthy",
            "daily_contacts": random.randint(5, 20),
            "days_infected": 0,
            "early_detect": early_detect
        }

        individuals.append(person)
        coords.append((lat, lon))

    print("✅ Population generated")
    
    with open("synthetic_population.json", "w") as f:
        json.dump(individuals, f, indent=2)
    print("✅ Population data saved")

    np.save("population_coords.npy", np.array(coords))
    print("✅ Coordinates saved for adjacency generation")

    return individuals, coords

def build_adjacency_list_batched(coords, radius_km=RADIUS_KM):
    start_time = time.time()
    print(f"⏳ Building adjacency list using KDTree...")

    avg_lat = np.mean([c[0] for c in coords])
    radius_deg = radius_km / degree_to_km(avg_lat)

    num_batches = (len(coords) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🧩 Processing {num_batches} batches...")

    barrel_index = {}

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(coords))
        batch_coords = coords[batch_start:batch_end]

        tree = KDTree(coords)
        batch_adjacency = {}

        for i, (lat, lon) in enumerate(batch_coords):
            global_idx = batch_start + i
            indices = tree.query_ball_point((lat, lon), r=radius_deg)

            neighbors = []
            for j in indices:
                if global_idx != j:
                    d = haversine_km(lat, lon, coords[j][0], coords[j][1])
                    if d <= radius_km:
                        neighbors.append(j)

            batch_adjacency[str(global_idx)] = neighbors

        filename = f"adjacency_batch_{batch_start}_{batch_end - 1}.json"
        filepath = os.path.join(TEMP_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(batch_adjacency, f)

        barrel_index[filename] = {
            "start_id": batch_start,
            "end_id": batch_end - 1
        }

    with open(ADJ_INDEX_FILE, "w") as f:
        json.dump(barrel_index, f, indent=2)

    elapsed = time.time() - start_time
    print(f"✅ Adjacency list built in {elapsed:.2f} seconds")
    print(f"✅ Index written to {ADJ_INDEX_FILE}")

def merge_adjacency_batches():
    print(f"⏳ Merging adjacency batches...")

    with open(ADJ_INDEX_FILE, "r") as f:
        barrel_index = json.load(f)

    full_adjacency_list = {}

    for batch_file, _ in tqdm(barrel_index.items()):
        with open(os.path.join(TEMP_DIR, batch_file), "r") as f:
            batch_adjacency = json.load(f)
        full_adjacency_list.update(batch_adjacency)

    with open("adjacency_list.json", "w") as f:
        json.dump(full_adjacency_list, f, indent=2)

    print("✅ All batches merged into full adjacency list")
    print("✅ Full adjacency list saved as 'full_adjacency_list.json'")

def main():
    start = time.time()

    if os.path.exists("synthetic_population.json") and os.path.exists("population_coords.npy"):
        print("✅ Population data already exists. Skipping generation.")
        with open("synthetic_population.json", "r") as f:
            individuals = json.load(f)
        coords = np.load("population_coords.npy").tolist()
    else:
        individuals, coords = generate_population()

    if os.path.exists(ADJ_INDEX_FILE):
        print("✅ Adjacency index already exists. Skipping adjacency generation.")
    else:
        build_adjacency_list_batched(coords)

    merge_adjacency_batches()
    total = time.time() - start
    print(f"✅ All processing completed in {total:.2f} seconds")

if __name__ == "__main__":
    main()
