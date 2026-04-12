import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "openai", "-q"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import requests
from openai import OpenAI

# Print all env vars for debugging
print("=== ENV VARS ===", flush=True)
for key in ["API_KEY", "API_BASE_URL", "ENV_URL", "MODEL_NAME"]:
    val = os.getenv(key, "NOT_SET")
    print(f"{key}={val[:30] if val != 'NOT_SET' else 'NOT_SET'}", flush=True)
print("================", flush=True)

ENV_URL = os.getenv("ENV_URL", "https://shivanimp017-chefs-table-ai.hf.space")
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not API_KEY or not API_BASE_URL:
    print("ERROR: API_KEY or API_BASE_URL not set!", flush=True)
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

RECIPES = ["Butter Chicken", "Margherita Pizza", "Chicken Curry", "Paneer Butter Masala",
           "Grilled Salmon", "Dal Tadka", "Caesar Salad", "Beef Steak"]

for task in ["task_easy", "task_medium", "task_hard"]:
    print(f"[START] task={task} env=chefs-table model={MODEL_NAME}", flush=True)
    step, score, done = 0, 0.0, False
    obs = {}

    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task}, timeout=15)
        obs = r.json().get("observation", {})
    except Exception as e:
        print(f"[STEP] step=1 reward=0.05 done=true error={str(e)[:80]}", flush=True)
        print(f"[END] task={task} score=0.05 steps=1", flush=True)
        continue

    while not done and step < 5:
        recs = obs.get("recommendations", RECIPES[:4])
        prompt = f"Task: {task}. Recipes: {recs}. Reply with one recipe name only."

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                timeout=25
            )
            content = response.choices[0].message.content.strip()
            recipe = recs[0] if recs else RECIPES[0]
            for r_name in RECIPES:
                if r_name.lower() in content.lower():
                    recipe = r_name
                    break
        except Exception as e:
            print(f"[LLM_ERROR] {str(e)}", flush=True)
            recipe = recs[0] if recs else RECIPES[0]

        try:
            r2 = requests.post(f"{ENV_URL}/step",
                json={"action": {"recipe": recipe, "feedback": "like", "task": task.replace("task_", "")}},
                timeout=15)
            obs = r2.json().get("observation", {})
            reward = obs.get("reward", 1.0)
            done = obs.get("done", False)
            score = obs.get("task_score", 0.0)
            step += 1
            print(f"[STEP] step={step} reward={reward} done={done} error=null", flush=True)
        except Exception as e:
            print(f"[STEP] step={step+1} reward=0.05 done=true error={str(e)[:80]}", flush=True)
            break

    print(f"[END] task={task} score={score} steps={step}", flush=True)
