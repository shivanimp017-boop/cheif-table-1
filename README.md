# 🍽️ Chef's Table — AI Recipe Assistant

> Built for AI Hackathon 2025 — Theme: **Building AI Environment using Reinforcement Learning**

## 🌐 Live Demo
👉 **https://huggingface.co/spaces/shivanimp017/chefs-table-ai**

## 🚀 Features
- 🔐 User Authentication (Register/Login)
- 🔍 Smart Recipe Search (TheMealDB API)
- 📷 Food Detection via Camera/Upload
- 🥦 Veg / 🍖 Non-Veg / ⭐ Favourites
- 👤 User Profile with Search History
- ➕ Add Your Own Recipes
- 🤖 AI-powered Recommendations (Q-Learning RL Agent)

## 🧠 Reinforcement Learning
The app uses a **Q-Learning agent** to recommend recipes:
- **State** — User's search history + favourites
- **Action** — Which recipe to recommend
- **Reward** — +1 if user likes, -0.5 if user dislikes
- **Policy** — Epsilon-greedy (80% exploit, 20% explore)

The agent learns and improves recommendations with every interaction!

## 🛠️ Tech Stack
- Python 3.14 + Flask
- TheMealDB API
- HTML / CSS / JavaScript
- Reinforcement Learning (Q-Learning)
- Hugging Face Spaces (Deployment)

## ⚙️ Setup
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`

## 📁 Project Structure
```
chefs-table-ai/
├── app.py              # Flask backend
├── rl_agent.py         # Q-Learning RL agent
├── requirements.txt
├── templates/
│   ├── index.html      # Landing page
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html  # Home with RL recommendations
│   ├── recipe.html     # Recipe detail page
│   ├── veg.html
│   ├── nonveg.html
│   ├── favourites.html
│   ├── profile.html
│   ├── detect.html     # Camera/Upload food detection
│   └── add_recipe.html
```

## 👩‍💻 Built by
**Shivani** — AI Hackathon 2025
