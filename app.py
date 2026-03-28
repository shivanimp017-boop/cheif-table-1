from flask import Flask, render_template, request, redirect, url_for, session
import os, json

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
    return render_template('dashboard.html', username=session['username'])

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

        def add_meals(meal_list, full=True):
            for meal in meal_list:
                if meal['idMeal'] not in seen:
                    if not full:
                        rd = requests.get(f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal['idMeal']}", timeout=5).json()
                        if rd.get('meals'):
                            meals.append(rd['meals'][0])
                            seen.add(meal['idMeal'])
                    else:
                        meals.append(meal)
                        seen.add(meal['idMeal'])

        # 1. Search by meal name
        r1 = requests.get(f'https://www.themealdb.com/api/json/v1/1/search.php?s={requests.utils.quote(query)}', timeout=5).json()
        if r1.get('meals'): add_meals(r1['meals'])

        # 2. Search by category
        r2 = requests.get(f'https://www.themealdb.com/api/json/v1/1/filter.php?c={requests.utils.quote(query)}', timeout=5).json()
        if r2.get('meals'): add_meals(r2['meals'][:6], full=False)

        # 3. Search by area/cuisine
        r3 = requests.get(f'https://www.themealdb.com/api/json/v1/1/filter.php?a={requests.utils.quote(query)}', timeout=5).json()
        if r3.get('meals'): add_meals(r3['meals'][:6], full=False)

        if not meals:
            return {'items': []}

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
        return {'items': results}
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
    import base64, requests
    data = request.get_json()
    image_data = data.get('image', '')
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        # Use Clarifai food recognition model (free)
        headers = { 'Authorization': 'Key YOUR_CLARIFAI_KEY', 'Content-Type': 'application/json' }
        payload = {
            'inputs': [{
                'data': { 'image': { 'base64': image_data } }
            }]
        }
        r = requests.post('https://api.clarifai.com/v2/models/bd367be194cf45149e75f01d59f77ba7/outputs', json=payload, headers=headers, timeout=10)
        result = r.json()
        concepts = result['outputs'][0]['data']['concepts']
        keywords = [c['name'] for c in concepts[:5] if c['value'] > 0.7]
        description = f"Detected: {', '.join(keywords[:3])}" if keywords else 'Could not identify food.'
        return jsonify({'description': description, 'keywords': keywords})
    except Exception as e:
        # Fallback: return generic message
        return jsonify({'description': 'Image received. Try searching manually.', 'keywords': ['chicken', 'pasta', 'salad']})

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
