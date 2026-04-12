import json
import os
import random
from datetime import datetime

RECIPES = [
    {'name': 'Grilled Salmon',       'id': '52959', 'img': 'https://www.themealdb.com/images/media/meals/xxyupu1468262513.jpg', 'category': 'nonveg', 'cuisine': 'seafood',  'meal_time': ['lunch', 'dinner']},
    {'name': 'Pasta Carbonara',      'id': '52982', 'img': 'https://www.themealdb.com/images/media/meals/llcbn01574260722.jpg', 'category': 'nonveg', 'cuisine': 'italian',  'meal_time': ['lunch', 'dinner']},
    {'name': 'Beef Steak',           'id': '52874', 'img': 'https://www.themealdb.com/images/media/meals/sytuqu1511553755.jpg', 'category': 'nonveg', 'cuisine': 'grill',    'meal_time': ['dinner']},
    {'name': 'Caesar Salad',         'id': '53049', 'img': 'https://www.themealdb.com/images/media/meals/xxrxux1503070723.jpg', 'category': 'veg',    'cuisine': 'salad',    'meal_time': ['lunch']},
    {'name': 'Chicken Curry',        'id': '52772', 'img': 'https://www.themealdb.com/images/media/meals/wyxwsp1486979827.jpg', 'category': 'nonveg', 'cuisine': 'indian',   'meal_time': ['lunch', 'dinner']},
    {'name': 'Chocolate Lava Cake',  'id': '52905', 'img': 'https://www.themealdb.com/images/media/meals/twspvx1511784937.jpg', 'category': 'veg',    'cuisine': 'dessert',  'meal_time': ['dinner']},
    {'name': 'Sushi Platter',        'id': '53065', 'img': 'https://www.themealdb.com/images/media/meals/g046bb1663960946.jpg', 'category': 'nonveg', 'cuisine': 'japanese', 'meal_time': ['lunch', 'dinner']},
    {'name': 'Margherita Pizza',     'id': '52768', 'img': 'https://www.themealdb.com/images/media/meals/x0lk931587671540.jpg', 'category': 'veg',    'cuisine': 'italian',  'meal_time': ['lunch', 'dinner']},
    {'name': 'Butter Chicken',       'id': '52869', 'img': 'https://www.themealdb.com/images/media/meals/tkxquw1628771028.jpg', 'category': 'nonveg', 'cuisine': 'indian',   'meal_time': ['lunch', 'dinner']},
    {'name': 'Dal Tadka',            'id': '52785', 'img': 'https://www.themealdb.com/images/media/meals/wuxrtu1483564410.jpg', 'category': 'veg',    'cuisine': 'indian',   'meal_time': ['lunch', 'dinner']},
    {'name': 'Paneer Butter Masala', 'id': '52807', 'img': 'https://www.themealdb.com/images/media/meals/1548772327.jpg',      'category': 'veg',    'cuisine': 'indian',   'meal_time': ['lunch', 'dinner']},
    {'name': 'Veg Noodles',          'id': '52772', 'img': 'https://www.themealdb.com/images/media/meals/wvpsxx1468256321.jpg', 'category': 'veg',    'cuisine': 'asian',    'meal_time': ['lunch', 'dinner']},
    {'name': 'Garlic Shrimp',        'id': '52819', 'img': 'https://www.themealdb.com/images/media/meals/xxpqsy1511452222.jpg', 'category': 'nonveg', 'cuisine': 'seafood',  'meal_time': ['dinner']},
    {'name': 'Chicken Tacos',        'id': '52830', 'img': 'https://www.themealdb.com/images/media/meals/ypxvwv1505333929.jpg', 'category': 'nonveg', 'cuisine': 'mexican',  'meal_time': ['lunch', 'dinner']},
    {'name': 'Falafel Wrap',         'id': '53091', 'img': 'https://www.themealdb.com/images/media/meals/ae6clc1760524712.jpg', 'category': 'veg',    'cuisine': 'middleeastern', 'meal_time': ['lunch']},
    {'name': 'Veggie Burger',        'id': '52829', 'img': 'https://www.themealdb.com/images/media/meals/xutquv1505330523.jpg', 'category': 'veg',    'cuisine': 'american', 'meal_time': ['lunch', 'dinner']},
]

RECIPE_NAMES = [r['name'] for r in RECIPES]
CATEGORIES   = {r['name']: r['category'] for r in RECIPES}
IMAGES       = {r['name']: r['img'] for r in RECIPES}
MEAL_IDS     = {r['name']: r['id'] for r in RECIPES}
CUISINES     = {r['name']: r['cuisine'] for r in RECIPES}
MEAL_TIMES   = {r['name']: r['meal_time'] for r in RECIPES}

RL_FILE = 'rl_data.json'

def load_rl_data():
    if os.path.exists(RL_FILE):
        with open(RL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_rl_data(data):
    with open(RL_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_meal_time():
    hour = datetime.now().hour
    if 6 <= hour < 11:  return 'breakfast'
    if 11 <= hour < 16: return 'lunch'
    return 'dinner'

def get_q_table(username):
    data = load_rl_data()
    if username not in data:
        data[username] = {
            'q_table': {r: 0.0 for r in RECIPE_NAMES},
            'interactions': 0,
            'veg_score': 0,
            'nonveg_score': 0,
        }
        save_rl_data(data)
    # Support old format
    if isinstance(data[username], dict) and 'q_table' not in data[username]:
        old = data[username]
        data[username] = {
            'q_table': old,
            'interactions': 0,
            'veg_score': 0,
            'nonveg_score': 0,
        }
        save_rl_data(data)
    return data[username]['q_table']

def update_reward(username, recipe, reward):
    """
    Q-Learning update:
    Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_Q - Q(s,a))
    """
    alpha = 0.3   # learning rate
    gamma = 0.9   # discount factor
    decay = 0.95  # decay old values slightly

    data = load_rl_data()
    if username not in data or 'q_table' not in data.get(username, {}):
        get_q_table(username)
        data = load_rl_data()

    q = data[username]['q_table']

    # Apply decay to all values (recent feedback matters more)
    for r in q:
        q[r] *= decay

    # Update the specific recipe
    if recipe in q:
        max_q = max(q.values())
        q[recipe] = q[recipe] + alpha * (reward + gamma * max_q - q[recipe])

    # Update category preference
    cat = CATEGORIES.get(recipe, '')
    if reward > 0:
        if cat == 'veg':
            data[username]['veg_score'] = data[username].get('veg_score', 0) + 1
        else:
            data[username]['nonveg_score'] = data[username].get('nonveg_score', 0) + 1

    data[username]['interactions'] = data[username].get('interactions', 0) + 1
    data[username]['q_table'] = q
    save_rl_data(data)

def get_recommendations(username, user_data=None, top_n=4):
    """
    Epsilon-greedy policy with:
    - Time-based filtering (meal time)
    - Category preference boost
    - Search history boost
    - Favourites boost
    """
    data = load_rl_data()
    if username not in data or 'q_table' not in data.get(username, {}):
        get_q_table(username)
        data = load_rl_data()

    q = dict(data[username]['q_table'])
    interactions = data[username].get('interactions', 0)
    veg_score    = data[username].get('veg_score', 0)
    nonveg_score = data[username].get('nonveg_score', 0)

    # Epsilon decreases as user interacts more (learns faster over time)
    epsilon = max(0.05, 0.3 - (interactions * 0.01))

    meal_time = get_meal_time()

    # Boost based on meal time
    for recipe in RECIPE_NAMES:
        if meal_time in MEAL_TIMES.get(recipe, []):
            q[recipe] = q.get(recipe, 0) + 0.2

    # Boost based on category preference
    if veg_score > nonveg_score:
        for recipe in RECIPE_NAMES:
            if CATEGORIES.get(recipe) == 'veg':
                q[recipe] = q.get(recipe, 0) + 0.3
    elif nonveg_score > veg_score:
        for recipe in RECIPE_NAMES:
            if CATEGORIES.get(recipe) == 'nonveg':
                q[recipe] = q.get(recipe, 0) + 0.3

    # Boost from favourites and search history
    if user_data:
        for recipe in RECIPE_NAMES:
            if recipe in user_data.get('favourites', []):
                q[recipe] = q.get(recipe, 0) + 0.8
            for h in user_data.get('search_history', []):
                if h.lower() in recipe.lower():
                    q[recipe] = q.get(recipe, 0) + 0.4

    # Epsilon-greedy selection
    if random.random() < epsilon:
        recommended = random.sample(RECIPE_NAMES, min(top_n, len(RECIPE_NAMES)))
    else:
        sorted_recipes = sorted(q.items(), key=lambda x: x[1], reverse=True)
        recommended = [r[0] for r in sorted_recipes[:top_n]]

    return [
        {
            'name':     r,
            'category': CATEGORIES.get(r, ''),
            'score':    round(data[username]['q_table'].get(r, 0), 2),
            'image':    IMAGES.get(r, ''),
            'meal_id':  MEAL_IDS.get(r, ''),
            'meal_time': meal_time,
        }
        for r in recommended
    ]
