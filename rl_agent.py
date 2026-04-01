import json
import os
import random

RECIPES = [
    {'name': 'Grilled Salmon',       'id': '52959', 'img': 'https://www.themealdb.com/images/media/meals/xxyupu1468262513.jpg', 'category': 'nonveg'},
    {'name': 'Pasta Carbonara',      'id': '52982', 'img': 'https://www.themealdb.com/images/media/meals/llcbn01574260722.jpg', 'category': 'nonveg'},
    {'name': 'Beef Steak',           'id': '52874', 'img': 'https://www.themealdb.com/images/media/meals/sytuqu1511553755.jpg', 'category': 'nonveg'},
    {'name': 'Caesar Salad',         'id': '53049', 'img': 'https://www.themealdb.com/images/media/meals/xxrxux1503070723.jpg', 'category': 'veg'},
    {'name': 'Chicken Curry',        'id': '52772', 'img': 'https://www.themealdb.com/images/media/meals/wyxwsp1486979827.jpg', 'category': 'nonveg'},
    {'name': 'Chocolate Lava Cake',  'id': '52893', 'img': 'https://www.themealdb.com/images/media/meals/joxuw61511793650.jpg', 'category': 'veg'},
    {'name': 'Sushi Platter',        'id': '53065', 'img': 'https://www.themealdb.com/images/media/meals/g046bb1663960946.jpg', 'category': 'nonveg'},
    {'name': 'Margherita Pizza',     'id': '52768', 'img': 'https://www.themealdb.com/images/media/meals/x0lk931587671540.jpg', 'category': 'veg'},
    {'name': 'Butter Chicken',       'id': '52869', 'img': 'https://www.themealdb.com/images/media/meals/tkxquw1628771028.jpg', 'category': 'nonveg'},
    {'name': 'Dal Tadka',            'id': '52785', 'img': 'https://www.themealdb.com/images/media/meals/wuxrtu1483564410.jpg', 'category': 'veg'},
    {'name': 'Paneer Butter Masala', 'id': '52807', 'img': 'https://www.themealdb.com/images/media/meals/1548772327.jpg',      'category': 'veg'},
    {'name': 'Veg Noodles',          'id': '52772', 'img': 'https://www.themealdb.com/images/media/meals/wvpsxx1468256321.jpg', 'category': 'veg'},
    {'name': 'Garlic Shrimp',        'id': '52819', 'img': 'https://www.themealdb.com/images/media/meals/xxpqsy1511452222.jpg', 'category': 'nonveg'},
    {'name': 'Chicken Tacos',        'id': '52818', 'img': 'https://www.themealdb.com/images/media/meals/cpkvms1511206747.jpg', 'category': 'nonveg'},
    {'name': 'Falafel Wrap',         'id': '52952', 'img': 'https://www.themealdb.com/images/media/meals/y3uxpq1487348960.jpg', 'category': 'veg'},
    {'name': 'Veggie Burger',        'id': '52829', 'img': 'https://www.themealdb.com/images/media/meals/xutquv1505330523.jpg', 'category': 'veg'},
]

RECIPE_NAMES = [r['name'] for r in RECIPES]
CATEGORIES   = {r['name']: r['category'] for r in RECIPES}
IMAGES       = {r['name']: r['img'] for r in RECIPES}
MEAL_IDS     = {r['name']: r['id'] for r in RECIPES}

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
        data[username] = {r: 0.0 for r in RECIPE_NAMES}
        save_rl_data(data)
    return data[username]

def update_reward(username, recipe, reward):
    alpha = 0.3
    data = load_rl_data()
    if username not in data:
        data[username] = {r: 0.0 for r in RECIPE_NAMES}
    if recipe in data[username]:
        old_q = data[username][recipe]
        data[username][recipe] = old_q + alpha * (reward - old_q)
    save_rl_data(data)

def get_recommendations(username, user_data=None, top_n=4):
    epsilon = 0.2
    q_table = get_q_table(username)

    if user_data:
        favourites = user_data.get('favourites', [])
        history    = user_data.get('search_history', [])
        for recipe in RECIPE_NAMES:
            if recipe in favourites:
                q_table[recipe] = max(q_table[recipe], 0.8)
            for h in history:
                if h.lower() in recipe.lower():
                    q_table[recipe] = max(q_table[recipe], 0.5)

    if random.random() < epsilon:
        recommended = random.sample(RECIPE_NAMES, min(top_n, len(RECIPE_NAMES)))
    else:
        sorted_recipes = sorted(q_table.items(), key=lambda x: x[1], reverse=True)
        recommended = [r[0] for r in sorted_recipes[:top_n]]

    return [
        {
            'name':     r,
            'category': CATEGORIES.get(r, ''),
            'score':    round(q_table.get(r, 0), 2),
            'image':    IMAGES.get(r, ''),
            'meal_id':  MEAL_IDS.get(r, '')
        }
        for r in recommended
    ]
