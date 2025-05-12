# simulation_utils.py
import random

def calculate_infection_probability(n, f, person, local_density, base_prob=0.05):
    """
    Calculate infection probability based on:
    - n: number of infected neighbors
    - f: frequency multiplier
    - person: full dict with age_group, chronic_illness, etc.
    - local_density: people nearby
    - base_prob: base chance of infection
    """
    age_risk_factor = {
        "Child": 0.8,
        "Adult": 1.0,
        "Elderly": 1.5
    }.get(person.get("age_group", "Adult"), 1.0)

    chronic_risk = 1.4 if person.get("chronic_illness", False) else 1.0
    distancing_effect = 0.6 if person.get("distancing", False) else 1.0
    contact_factor = min(person.get("daily_contacts", 10) / 10.0, 2.0)
    late_detection = 1.2 if person.get("early_detect", 1.0) > 0.5 else 1.0

    density_factor = min(local_density / 50.0, 2.0)
    adjusted_base = base_prob * density_factor * age_risk_factor * chronic_risk * contact_factor * late_detection

    chance = 1 - (1 - adjusted_base) ** (n * f)
    chance *= distancing_effect

    return min(chance, 0.98)



def determine_recovery_or_death(person, days_infected):
    if days_infected >= 30:
        return "deceased" if random.random() < 0.9 else "recovered"

    if person["age_group"] == "Child":
        return "recovered" if days_infected >= 14 else "infected"
    elif person["age_group"] == "Adult" and not person["chronic_illness"]:
        return "recovered" if days_infected >= 5 else "infected"
    else:
        return "recovered" if days_infected >= 20 else "infected"
