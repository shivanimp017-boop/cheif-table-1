import json
import os
import random

RECIPES = [
    "Grilled Salmon", "Pasta Carbonara", "Beef Steak", "Caesar Salad",
    "Chicken Curry", "Chocolate Lava Cake", "Sushi Platter", "Margherita Pizza",
    "Butter Chicken", "Dal Tadka", "Paneer Butter Masala", "Veg Noodles",
    "Garlic Shrimp", "Chicken Tacos", "Falafel Wrap", "Veggie Burger"
]

CATEGORIES = {
    "Grilled Salmon": "nonveg", "Pasta Carbonara": "nonveg", "Beef Steak": "nonveg",
    "Caesar Salad": "veg", "Chicken Curry": "nonveg", "Chocolate Lava Cake": "veg",
    "Sushi Platter": "nonveg", "Margherita Pizza": "veg", "Butter Chicken": "nonveg",
    "Dal Tadka": "veg", "Paneer Butter Masala": "veg", "Veg Noodles": "veg",
    "Garlic Shrimp": "nonveg", "Chicken Tacos": "nonveg", "Falafel Wrap": "veg",
    "Veggie Burger": "veg"
}

RL_FILE = 'rl_data.json'

def load_rl_data():
    if os.path.exists(RL_FILE):
        with open(RL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_rl_data(data):
    with open(RL_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_q_table(username):
    data = load_rl_data()
    if username not in data:
        # Initialize Q-table with 0 values for all recipes
        data[username] = {recipe: 0.0 for recipe in RECIPES}
        save_rl_data(data)
    return data[username]

def update_reward(username, recipe, reward):
    """
    Update Q-value using Q-Learning formula:
    Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
    """
    alpha = 0.3  # learning rate
    data = load_rl_data()
    if username not in data:
        data[username] = {r: 0.0 for r in RECIPES}
    if recipe in data[username]:
        old_q = data[username][recipe]
        data[username][recipe] = old_q + alpha * (reward - old_q)
    save_rl_data(data)

def get_recommendations(username, user_data=None, top_n=4):
    """
    Get top N recipe recommendations using epsilon-greedy policy.
    Considers user favourites and search history as prior knowledge.
    """
    epsilon = 0.2  # 20% exploration, 80% exploitation
    q_table = get_q_table(username)

    # Boost Q-values based on favourites and search history
    if user_data:
        favourites = user_data.get('favourites', [])
        history    = user_data.get('search_history', [])
        for recipe in RECIPES:
            if recipe in favourites:
                q_table[recipe] = max(q_table[recipe], 0.8)
            for h in history:
                if h.lower() in recipe.lower():
                    q_table[recipe] = max(q_table[recipe], 0.5)

    if random.random() < epsilon:
        # Explore: pick random recipes
        recommended = random.sample(RECIPES, min(top_n, len(RECIPES)))
    else:
        # Exploit: pick highest Q-value recipes
        sorted_recipes = sorted(q_table.items(), key=lambda x: x[1], reverse=True)
        recommended = [r[0] for r in sorted_recipes[:top_n]]

    return [{'name': r, 'category': CATEGORIES.get(r, ''), 'score': round(q_table.get(r, 0), 2)} for r in recommended]
