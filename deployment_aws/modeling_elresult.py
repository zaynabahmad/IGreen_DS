import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
gamma_base = 0.8
rho_base = 0.1
M_max_per_area = 1500
area = 1
M_max = M_max_per_area * area
init_m = 10

# Genetic Algorithm Parameters
POPULATION_SIZE = 150
GENOME_LENGTH = 3  # [lux, T, water_supply]
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 150

stage_factors = {
    "Germination": 0.6,
    "Vegetative": 1.0,
    "Flowering": .90,
}

stages = {
    "Germination": {"lux": (0, 40000), "temp": (20, 30), "water": (0.6, 1.0), "red_light": 0.6, "blue_light": 0.4},
    "Vegetative": {"lux": (10000, 50000), "temp": (20, 30), "water": (0.3, 0.9), "red_light": 0.5, "blue_light": 0.5},
    "Flowering": {"lux": (40000, 60000), "temp": (22, 28), "water": (0.2, 0.7), "red_light": 0.7, "blue_light": 0.3},
}

# Function to calculate photosynthesis rate
def photosynthesis_rate(lux, T, water_supply, stage_factor, red_light, blue_light):
    light_saturation_point = 54000  # lavender = 54000
    light_factor = lux / (lux + light_saturation_point)
    
    # Influence of red and blue light
    light_color_factor = (red_light * 0.7) + (blue_light * 0.3)  # Adjust this factor based on experiments or literature
    
    temperature_factor = np.exp(-0.1 * (T - 20) ** 2)
    water_factor = water_supply
    return gamma_base * light_factor * light_color_factor * temperature_factor * water_factor * stage_factor


# Function to calculate respiration rate
def respiration_rate(lux, T):
    return rho_base * (1 + 0.3 * np.cos(lux / 10000)) * np.exp(-0.15 * (T - 22) ** 2)

# Function to calculate plant growth rate
# Modify plant_growth_rate to include red_light and blue_light
def plant_growth_rate(M, lux, T, water_supply, stage_factor, red_light, blue_light):
    light_saturation_point = 54000  # lavender = 54000
    red_blue_factor = 0.45 * red_light + 0.55 * blue_light  # Example of combining red and blue light
    light_factor = lux / (lux + light_saturation_point) * red_blue_factor  # Modify light factor with red/blue influence
    temperature_factor = np.exp(-0.1 * (T - 20) ** 2)
    water_factor = water_supply
    r = gamma_base * light_factor * temperature_factor * water_factor * stage_factor
    respiration = respiration_rate(lux, T)
    growth_rate = (r * (M_max - M) * M / M_max) - respiration
    growth_rate = max(growth_rate, 0)  # Ensure growth rate doesn't go negative
    return growth_rate


def optimize_conditions(stage_name, stage_ranges):
    population_size = POPULATION_SIZE
    generations = GENERATIONS

    # Initialize population
    population = [
        {
            "lux": random.uniform(*stage_ranges["lux"]),
            "temp": random.uniform(*stage_ranges["temp"]),
            "water": random.uniform(*stage_ranges["water"]),
        }
        for _ in range(population_size)
    ]

    for generation in range(generations):
        # Update fitness calculation to include red_light and blue_light
        fitness_scores = [
            plant_growth_rate(init_m, individual["lux"], individual["temp"], individual["water"], stage_factors.get(stage_name, 1.0), 
                              stage_ranges.get("red_light", 0), stage_ranges.get("blue_light", 0))
            for individual in population
        ]

        min_fitness = min(fitness_scores)
        max_fitness = max(fitness_scores)

        normalized_fitness_scores = [
            (fitness - min_fitness) / (max_fitness - min_fitness) if max_fitness != min_fitness else 1.0
            for fitness in fitness_scores
        ]

        top_individuals = sorted(
            zip(population, normalized_fitness_scores), key=lambda x: x[1], reverse=True
        )[:population_size // 2]

        # Crossover and Mutation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.choices(top_individuals, k=2)
            child = {}
            for key in ["lux", "temp", "water"]:
                if random.random() < CROSSOVER_RATE:
                    child[key] = parent1[0][key]
                else:
                    child[key] = parent2[0][key]

            # Mutation
            if random.random() < MUTATION_RATE:
                child["lux"] += random.uniform(-500, 500)
                child["lux"] = max(stage_ranges["lux"][0], min(child["lux"], stage_ranges["lux"][1]))
            if random.random() < MUTATION_RATE:
                child["temp"] += random.uniform(-1.5, 1.5)
                child["temp"] = max(stage_ranges["temp"][0], min(child["temp"], stage_ranges["temp"][1]))
            if random.random() < MUTATION_RATE:
                child["water"] += random.uniform(-0.05, 0.05)
                child["water"] = max(stage_ranges["water"][0], min(child["water"], stage_ranges["water"][1]))

            # Ensure the mutated values stay within range
            child["lux"] = max(stage_ranges["lux"][0], min(child["lux"], stage_ranges["lux"][1]))
            child["temp"] = max(stage_ranges["temp"][0], min(child["temp"], stage_ranges["temp"][1]))
            child["water"] = max(stage_ranges["water"][0], min(child["water"], stage_ranges["water"][1]))

            new_population.append(child)

        population = new_population

    # Selection
    best_individual = max(population, key=lambda ind: plant_growth_rate(
        init_m, 
        ind["lux"], 
        ind["temp"], 
        ind["water"], 
        stage_factors.get(stage_name, 1.0),
        stage_ranges.get("red_light", 0),  # Add red_light here
        stage_ranges.get("blue_light", 0)   # Add blue_light here
    ))
    best_individual["red_light"] = stage_ranges["red_light"]
    best_individual["blue_light"] = stage_ranges["blue_light"]
    return best_individual

# Optimize conditions for each stage
optimized_conditions = {}
for stage, ranges in stages.items():
    print(f"Optimizing conditions for {stage}...")
    optimized_conditions[stage] = optimize_conditions(stage, ranges)

# Print optimized conditions for each stage
for stage, conditions in optimized_conditions.items():
    print(f"Optimal conditions for {stage}: {conditions}")

# Simulate and plot biomass growth and growth rate for best conditions
M = init_m  # Initialize mass for the stage
for stage, conditions in optimized_conditions.items():
    print(f"Simulating growth for {stage} stage...")
    stage_factor = stage_factors.get(stage, 1.0)

    time_steps = 30
    biomass_data = []
    growth_rate_data = []

    for t in range(time_steps):
        growth_rate = plant_growth_rate(M, conditions["lux"], conditions["temp"], conditions["water"], stage_factor, 
                                        conditions["red_light"], conditions["blue_light"])
        M += growth_rate

        M = min(max(M, 0), M_max)

        biomass_data.append(M)
        growth_rate_data.append(growth_rate)

    # Plot biomass and growth rate
    plt.figure(figsize=(12, 6))

    # Biomass plot
    plt.subplot(1, 2, 1)
    plt.plot(range(time_steps), biomass_data, label="Biomass", color="blue")
    plt.title(f"Biomass Growth - {stage} Stage")
    plt.xlabel("Time (Steps)")
    plt.ylabel("Biomass (grams)")
    plt.grid(True)

    # Growth rate plot
    plt.subplot(1, 2, 2)
    plt.plot(range(time_steps), growth_rate_data, label="Growth Rate", color="green")
    plt.title(f"Growth Rate - {stage} Stage")
    plt.xlabel("Time (Steps)")
    plt.ylabel("Growth Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final mass at the end of the stage
    print(f"Final biomass at the end of {stage} stage: {M:.2f} grams")
