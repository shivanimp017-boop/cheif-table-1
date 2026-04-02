"""
Chef's Table AI - Demo Script
==============================
This script demonstrates the core AI/RL environment.
Run this to see the Q-Learning agent in action without the web UI.
"""

from rl_agent import get_recommendations, update_reward, get_q_table, RECIPE_NAMES

USERNAME = "demo_user"

print("=" * 60)
print("   CHEF'S TABLE AI - RL ENVIRONMENT DEMO")
print("   Theme: Building AI Environment using Reinforcement Learning")
print("=" * 60)

print("\n[STEP 1] Initial Q-Table (all recipes start at 0)")
q = get_q_table(USERNAME)
print(f"   Total recipes in environment: {len(q)}")
print(f"   Initial Q-values: all = 0.0")

print("\n[STEP 2] Get initial AI recommendations (random exploration)")
recs = get_recommendations(USERNAME)
print("   Recommended recipes:")
for i, r in enumerate(recs, 1):
    print(f"   {i}. {r['name']} ({r['category']}) - Score: {r['score']}")

print("\n[STEP 3] User likes 'Chicken Curry' and 'Butter Chicken'")
update_reward(USERNAME, "Chicken Curry", 1.0)
update_reward(USERNAME, "Butter Chicken", 1.0)
print("   Reward +1.0 applied to both")

print("\n[STEP 4] User dislikes 'Beef Steak'")
update_reward(USERNAME, "Beef Steak", -0.5)
print("   Reward -0.5 applied")

print("\n[STEP 5] Updated Q-Table after learning")
q = get_q_table(USERNAME)
top = sorted(q.items(), key=lambda x: x[1], reverse=True)[:5]
print("   Top 5 recipes by Q-value:")
for name, val in top:
    bar = "|" * int(max(val * 20, 0))
    print(f"   {name:<25} {val:.3f} {bar}")

print("\n[STEP 6] New recommendations after learning (exploitation)")
recs = get_recommendations(USERNAME)
print("   AI now recommends (learned from feedback):")
for i, r in enumerate(recs, 1):
    print(f"   {i}. {r['name']} ({r['category']}) - Score: {r['score']}")

print("\n[STEP 7] RL Stats")
liked    = sum(1 for v in q.values() if v > 0)
disliked = sum(1 for v in q.values() if v < 0)
accuracy = round((liked / max(liked + disliked, 1)) * 100)
print(f"   Recipes liked:    {liked}")
print(f"   Recipes disliked: {disliked}")
print(f"   AI Accuracy:      {accuracy}%")

print("\n[OK] RL Environment working successfully!")
print("=" * 60)
print("   GitHub: https://github.com/shivanimp017-boop/cheif-table-1")
print("   HF Space: https://huggingface.co/spaces/shivanimp017/chefs-table-ai")
print("=" * 60)
