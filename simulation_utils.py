# simulation_utils.py
import random

def calculate_infection_probability(n, f, person, local_density, base_prob=0.08, infection_variant_factor=1.5):
    """
    Increased infection probability logic with more aggressive risk factors.
    """
    age_risk_factor = {
        "Child": 1.0,    # was 0.8
        "Adult": 1.2,    # was 1.0
        "Elderly": 2.0   # was 1.5
    }.get(person.get("age_group", "Adult"), 1.2)

    chronic_risk = 1.8 if person.get("chronic_illness", False) else 1.2  # was 1.4 / 1.0
    distancing_effect = 0.8 if person.get("distancing", False) else 1.0  # was 0.6 / 1.0 (less effective)
    contact_factor = min(person.get("daily_contacts", 10) / 7.0, 4.0)    # was /10.0 and capped at 3.0
    late_detection = 1.5 if person.get("early_detect", 1.0) > 0.5 else 1.2  # was 1.2 / 1.0

    environmental_factor = 1.5 if person.get("high_risk_area", False) else 1.2  # was 1.2 / 1.0
    mutation_factor = infection_variant_factor  # Increased default to 1.5 (was 1.2)

    density_factor = min(local_density / 30.0, 4.0)  # was /50.0 and capped at 3.0 (more sensitive to density)

    # Compute adjusted base probability with higher multipliers
    adjusted_base = base_prob * density_factor * age_risk_factor * chronic_risk * contact_factor * late_detection * environmental_factor * mutation_factor

    # Chance of infection through n contacts with factor f
    chance = 1 - (1 - adjusted_base) ** (n * f)
    chance *= distancing_effect

    # Cap the chance at 99.5%
    return min(chance, 0.995)


def determine_recovery_or_death(person, days_infected):
    if days_infected >= 30:
        # 90% chance of death after 30 days
        return "deceased" if random.random() < 0.9 else "recovered"

    age_group = person["age_group"]
    has_chronic = person["chronic_illness"]

    if age_group == "Child":
        if days_infected >= 14:
            # 60% chance to recover after 14 days
            return "recovered" if random.random() < 0.6 else "infected"
    elif age_group == "Adult" and not has_chronic:
        if days_infected >= 5:
            # 50% chance to recover after 5 days
            return "recovered" if random.random() < 0.5 else "infected"
    else:
        if days_infected >= 20:
            # 30% chance to recover after 20 days
            return "recovered" if random.random() < 0.3 else "infected"

    return "infected"
