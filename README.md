---
title: Chefs Table AI
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_file: inference.py
---

# 🍽️ Chef's Table — AI Recipe Assistant

> Built for AI Hackathon 2025 — Theme: **Building AI Environment using Reinforcement Learning**

## 🌐 Live Demo
👉 **https://huggingface.co/spaces/shivanimp017/chefs-table-ai**

## 🚀 Features
- 🔐 User Authentication (Register/Login)
- 🔍 Smart Recipe Search (TheMealDB API + Indian Recipes)
- 📷 Food Detection via Camera/Upload with Nutrition Info
- 🥦 Veg / 🍖 Non-Veg / ⭐ Favourites
- 👤 User Profile with Search History
- ➕ Add Your Own Recipes
- 🤖 AI-powered Recommendations (Q-Learning RL Agent)
- 🎯 Food Quiz to train the AI
- 📈 RL Stats Dashboard
- 🔔 Notifications

## 🧠 Reinforcement Learning Environment

The app implements a **Q-Learning RL environment** where:

### State
- User's search history
- Liked/disliked recipes
- Time of day (breakfast/lunch/dinner)
- Category preference (veg/non-veg)

### Action
- Which recipe to recommend from 16 options

### Reward
- **+1.0** if user likes a recipe
- **-0.5** if user dislikes a recipe

### Policy
- **Epsilon-greedy** — starts with 30% exploration, reduces as user interacts more
- **Decay** — recent feedback matters more than old feedback
- **Gamma discount** — considers future rewards

### Learning
The agent improves with every interaction using:
```
Q(s,a) = Q(s,a) + α * (reward + γ * max_Q - Q(s,a))
```
Where α=0.3 (learning rate), γ=0.9 (discount factor)

## 🤖 OpenEnv Environment Spec

### Tasks
| Task | Description | Max Steps | Target |
|------|-------------|-----------|--------|
| **Easy** | Like 3 unique recipes | 10 | 3 likes |
| **Medium** | Like 2 veg + 2 non-veg recipes | 15 | Balanced |
| **Hard** | Achieve total reward ≥ 5.0 | 20 | Score ≥ 5.0 |

### Action Space
```json
{"recipe": "string", "feedback": "like|dislike", "task": "easy|medium|hard"}
```

### Observation Space
```json
{"recommendations": ["list"], "q_values": {}, "step": 0, "total_reward": 0.0, "reward": 0.0, "done": false, "task": "easy", "task_score": 0.0, "task_progress": "string"}
```

### Reward Function
- **+1.0** like a recipe
- **+0.3** partial progress (new unique recipe liked)
- **-0.5** dislike a recipe

### Baseline Scores
```bash
python baseline.py
```

## 🛠️ Tech Stack
- Python 3.14 + Flask
- TheMealDB API
- Imagga API (food detection)
- HTML / CSS / JavaScript
- Reinforcement Learning (Q-Learning)
- Hugging Face Spaces (Deployment)

## ⚙️ Setup
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`

## 🎮 Demo Script
Run the RL environment demo without the web UI:
```bash
python demo.py
```

## 📁 Project Structure
```
chefs-table-ai/
├── app.py              # Flask backend
├── rl_agent.py         # Q-Learning RL agent
├── indian_recipes.py   # Indian recipe database
├── demo.py             # RL environment demo script
├── requirements.txt
├── static/
│   └── images/         # Local food images
└── templates/
    ├── index.html       # Landing page
    ├── login.html
    ├── register.html
    ├── dashboard.html   # Home with RL recommendations
    ├── recipe.html      # Recipe detail page
    ├── indian_recipe.html
    ├── veg.html
    ├── nonveg.html
    ├── favourites.html
    ├── profile.html
    ├── detect.html      # Camera/Upload food detection
    ├── add_recipe.html
    ├── quiz.html        # Food Quiz (trains RL agent)
    ├── rl_stats.html    # RL learning stats
    ├── notifications.html
    └── settings.html
```

## 👩‍💻 Built by
**Shivani** — AI Hackathon 2025
