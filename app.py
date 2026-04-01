from flask import Flask, render_template, request, redirect, url_for, session
import os, json
from rl_agent import get_recommendations, update_reward
from indian_recipes import INDIAN_RECIPES, get_indian_recipe

app = Flask(__name__)
app.secret_key = 'chefs_table_secret_key'

USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    users = load_users()
    username = session['username']
    user = users.get(username, {})
    recommendations = get_recommendations(username, user)
    return render_template('dashboard.html', username=username, recommendations=recommendations)

@app.route('/rl_feedback', methods=['POST'])
@login_required
def rl_feedback():
    from flask import jsonify
    recipe = request.form.get('recipe', '').strip()
    action = request.form.get('action', '')  # 'like' or 'dislike'
    if recipe:
        reward = 1.0 if action == 'like' else -0.5
        update_reward(session['username'], recipe, reward)
    return ('', 204)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        users = load_users()
        if username in users and users[username]['password'] == password:
            session.permanent = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm  = request.form['confirm_password']
        users = load_users()
        if username in users:
            error = 'Username already exists.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            users[username] = {
                'password': password,
                'name':     request.form.get('name', '').strip(),
                'email':    request.form.get('email', '').strip(),
                'phone':    request.form.get('phone', '').strip(),
                'dob':      request.form.get('dob', ''),
                'gender':   request.form.get('gender', ''),
            }
            save_users(users)
            session.permanent = True
            session['username'] = username
            return redirect(url_for('dashboard'))
    return render_template('register.html', error=error)

@app.route('/profile')
@login_required
def profile():
    users = load_users()
    username = session['username']
    user = users.get(username, {})
    search_history = user.get('search_history', [])
    favourites = user.get('favourites', [])
    my_recipes = user.get('my_recipes', [])
    return render_template('profile.html', username=username, user=user, search_history=search_history, favourites=favourites, my_recipes=my_recipes)

@app.route('/get_history')
@login_required
def get_history():
    from flask import jsonify
    users = load_users()
    username = session['username']
    history = users.get(username, {}).get('search_history', [])
    return jsonify({'history': history})

@app.route('/save_search', methods=['POST'])
@login_required
def save_search():
    query = request.form.get('query', '').strip()
    if query:
        users = load_users()
        username = session['username']
        history = users[username].get('search_history', [])
        if query not in history:
            history.insert(0, query)
        users[username]['search_history'] = history[:10]
        save_users(users)
    return ('', 204)

@app.route('/toggle_favourite', methods=['POST'])
@login_required
def toggle_favourite():
    item = request.form.get('item', '').strip()
    if item:
        users = load_users()
        username = session['username']
        favs = users[username].get('favourites', [])
        if item in favs:
            favs.remove(item)
        else:
            favs.insert(0, item)
        users[username]['favourites'] = favs
        save_users(users)
    return ('', 204)

@app.route('/indian_recipe/<recipe_key>')
@login_required
def indian_recipe(recipe_key):
    recipe = INDIAN_RECIPES.get(recipe_key)
    if not recipe:
        return redirect(url_for('dashboard'))
    return render_template('indian_recipe.html', recipe=recipe)

@app.route('/recipe/<meal_id>')
@login_required
def recipe(meal_id):
    import requests
    r = requests.get(f'https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal_id}')
    data = r.json()
    meal = data['meals'][0] if data['meals'] else None
    ingredients = []
    if meal:
        for i in range(1, 21):
            ing  = meal.get(f'strIngredient{i}', '').strip()
            meas = meal.get(f'strMeasure{i}', '').strip()
            if ing:
                ingredients.append({'name': ing, 'measure': meas})
    return render_template('recipe.html', meal=meal, ingredients=ingredients)

@app.route('/search')
@login_required
def search():
    import requests
    query = request.args.get('q', '').strip()
    if not query:
        return {'items': []}
    try:
        meals = []
        seen = set()
        results = []

        def add_meals(meal_list, full=True):
            for meal in meal_list:
                if meal['idMeal'] not in seen:
                    if not full:
                        rd = requests.get(f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal['idMeal']}", timeout=10).json()
                        if rd.get('meals'):
                            meals.append(rd['meals'][0])
                            seen.add(meal['idMeal'])
                    else:
                        meals.append(meal)
                        seen.add(meal['idMeal'])

        # 1. Search by meal name
        try:
            r1 = requests.get(f'https://www.themealdb.com/api/json/v1/1/search.php?s={requests.utils.quote(query)}', timeout=10).json()
            if r1.get('meals'): add_meals(r1['meals'])
        except: pass

        # 2. Search by category
        try:
            r2 = requests.get(f'https://www.themealdb.com/api/json/v1/1/filter.php?c={requests.utils.quote(query)}', timeout=10).json()
            if r2.get('meals'): add_meals(r2['meals'][:6], full=False)
        except: pass

        # 3. Search by area/cuisine
        try:
            r3 = requests.get(f'https://www.themealdb.com/api/json/v1/1/filter.php?a={requests.utils.quote(query)}', timeout=10).json()
            if r3.get('meals'): add_meals(r3['meals'][:6], full=False)
        except: pass

        if meals:
            results = []
            for meal in meals[:8]:
                results.append({
                    'id': meal['idMeal'],
                    'title': meal['strMeal'],
                    'category': meal.get('strCategory', ''),
                    'area': meal.get('strArea', ''),
                    'thumb': meal.get('strMealThumb', ''),
                    'instructions': (meal.get('strInstructions') or '')[:150] + '...',
                    'link': f"/recipe/{meal['idMeal']}"
                })

        # Always also check Indian recipes database and merge
        q = query.lower()
        seen_titles = {r['title'].lower() for r in results} if meals else set()
        for key, val in INDIAN_RECIPES.items():
            if key in q or q in key or q == key:
                if val['title'].lower() not in seen_titles:
                    results.append({
                        'id': key,
                        'title': val['title'],
                        'category': val['category'],
                        'area': val['area'],
                        'thumb': val['thumb'],
                        'instructions': val['instructions'][0] if val.get('instructions') else '',
                        'link': f"/indian_recipe/{key}"
                    })

        return {'items': results[:8] if results else []}

    except Exception as e:
        return {'items': [], 'error': str(e)}

@app.route('/add_recipe', methods=['GET', 'POST'])
@login_required
def add_recipe():
    success = False
    if request.method == 'POST':
        recipe = {
            'name':        request.form.get('name', '').strip(),
            'category':    request.form.get('category', ''),
            'cuisine':     request.form.get('cuisine', '').strip(),
            'prep_time':   request.form.get('prep_time', '').strip(),
            'cook_time':   request.form.get('cook_time', '').strip(),
            'serves':      request.form.get('serves', '').strip(),
            'ingredients': request.form.getlist('ingredients[]'),
            'steps':       request.form.get('steps', '').strip(),
            'notes':       request.form.get('notes', '').strip(),
            'added_by':    session['username']
        }
        users = load_users()
        username = session['username']
        if 'my_recipes' not in users[username]:
            users[username]['my_recipes'] = []
        users[username]['my_recipes'].insert(0, recipe)
        save_users(users)
        success = True
    return render_template('add_recipe.html', success=success)

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/detect_food', methods=['POST'])
@login_required
def detect_food():
    from flask import jsonify
    import requests, base64
    data = request.get_json()
    food_name = data.get('food_name', '').strip()
    image_data = data.get('image', '')

    # If image is provided, detect food from image first
    if image_data and not food_name:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

            # Use Imagga API for food detection
            api_key    = 'acc_e1ef3474774f3bc'
            api_secret = '7bfa1558c737b27a546c86f2c6a2a81f'
            response = requests.post(
                'https://api.imagga.com/v2/tags',
                auth=(api_key, api_secret),
                files={'image': ('food.jpg', image_bytes, 'image/jpeg')},
                timeout=10
            )
            result = response.json()
            tags = result.get('result', {}).get('tags', [])
            food_tags = [t['tag']['en'] for t in tags[:5] if t['confidence'] > 30]
            food_name = food_tags[0] if food_tags else 'food'
        except:
            food_name = 'food'

    if not food_name:
        return jsonify({'error': 'No food detected'})

    return jsonify({'food': food_name, 'nutrition': get_estimated_nutrition(food_name)})

def get_estimated_nutrition(food_name):
    """Estimated nutrition per 100g for common foods"""
    food = food_name.lower()
    db = {
        'chicken':  {'calories':165,'protein':31,'fat':3.6,'carbs':0,'fiber':0,'sugar':0,'calcium':15,'iron':1.0,'vitamin_c':0,'vitamin_a':9},
        'rice':     {'calories':130,'protein':2.7,'fat':0.3,'carbs':28,'fiber':0.4,'sugar':0,'calcium':10,'iron':0.2,'vitamin_c':0,'vitamin_a':0},
        'pasta':    {'calories':158,'protein':5.8,'fat':0.9,'carbs':31,'fiber':1.8,'sugar':0.6,'calcium':7,'iron':0.5,'vitamin_c':0,'vitamin_a':0},
        'salmon':   {'calories':208,'protein':20,'fat':13,'carbs':0,'fiber':0,'sugar':0,'calcium':12,'iron':0.3,'vitamin_c':3,'vitamin_a':12},
        'pizza':    {'calories':266,'protein':11,'fat':10,'carbs':33,'fiber':2.3,'sugar':3.6,'calcium':200,'iron':2.5,'vitamin_c':1,'vitamin_a':50},
        'salad':    {'calories':15, 'protein':1.3,'fat':0.2,'carbs':2.9,'fiber':1.8,'sugar':1.2,'calcium':36,'iron':0.9,'vitamin_c':9,'vitamin_a':370},
        'burger':   {'calories':295,'protein':17,'fat':14,'carbs':24,'fiber':1,'sugar':5,'calcium':100,'iron':2.7,'vitamin_c':1,'vitamin_a':20},
        'egg':      {'calories':155,'protein':13,'fat':11,'carbs':1.1,'fiber':0,'sugar':1.1,'calcium':56,'iron':1.8,'vitamin_c':0,'vitamin_a':160},
        'banana':   {'calories':89, 'protein':1.1,'fat':0.3,'carbs':23,'fiber':2.6,'sugar':12,'calcium':5,'iron':0.3,'vitamin_c':8.7,'vitamin_a':3},
        'apple':    {'calories':52, 'protein':0.3,'fat':0.2,'carbs':14,'fiber':2.4,'sugar':10,'calcium':6,'iron':0.1,'vitamin_c':4.6,'vitamin_a':3},
    }
    for key in db:
        if key in food:
            return db[key]
    return {'calories':200,'protein':8,'fat':7,'carbs':25,'fiber':2,'sugar':3,'calcium':50,'iron':1.5,'vitamin_c':5,'vitamin_a':20}

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    users = load_users()
    username = session['username']
    user = users.get(username, {})
    success = False
    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'profile':
            users[username]['name']  = request.form.get('name', '').strip()
            users[username]['email'] = request.form.get('email', '').strip()
            users[username]['phone'] = request.form.get('phone', '').strip()
            save_users(users)
            success = True
        elif action == 'password':
            current = request.form.get('current_password')
            new_pw  = request.form.get('new_password')
            confirm = request.form.get('confirm_password')
            if users[username]['password'] != current:
                error = 'Current password is incorrect.'
            elif new_pw != confirm:
                error = 'New passwords do not match.'
            else:
                users[username]['password'] = new_pw
                save_users(users)
                success = True
    return render_template('settings.html', user=users.get(username, {}), success=success, error=error)

@app.route('/notifications')
@login_required
def notifications():
    users = load_users()
    username = session['username']
    user = users.get(username, {})
    notifs = user.get('notifications', [])
    if not notifs:
        from datetime import datetime
        notifs = [
            {'icon': '🤖', 'title': 'RL Agent Active', 'message': 'Your AI agent is learning your food preferences!', 'time': 'Just now', 'unread': True},
            {'icon': '🍽️', 'title': 'Welcome to Chef Table!', 'message': 'Start searching recipes and liking them to train your AI.', 'time': 'Today', 'unread': True},
        ]
    return render_template('notifications.html', notifications=notifs)

@app.route('/clear_notifications', methods=['POST'])
@login_required
def clear_notifications():
    users = load_users()
    users[session['username']]['notifications'] = []
    save_users(users)
    return redirect(url_for('notifications'))

@app.route('/rl_stats')
@login_required
def rl_stats():
    from rl_agent import get_q_table
    username = session['username']
    q_table = get_q_table(username)
    q_scores = sorted(q_table.items(), key=lambda x: x[1], reverse=True)
    liked    = sum(1 for v in q_table.values() if v > 0)
    disliked = sum(1 for v in q_table.values() if v < 0)
    accuracy = round((liked / max(liked + disliked, 1)) * 100)
    return render_template('rl_stats.html', q_scores=q_scores, total_recipes=len(q_table), liked=liked, disliked=disliked, accuracy=accuracy)

@app.route('/quiz')
@login_required
def quiz():
    return render_template('quiz.html')

@app.route('/get_recommendations')
@login_required
def get_recs():
    from flask import jsonify
    users = load_users()
    username = session['username']
    user = users.get(username, {})
    recs = get_recommendations(username, user)
    return jsonify({'items': recs})

@app.route('/veg')
@login_required
def veg():
    return render_template('veg.html')

@app.route('/nonveg')
@login_required
def nonveg():
    return render_template('nonveg.html')

@app.route('/favourites')
@login_required
def favourites():
    return render_template('favourites.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
