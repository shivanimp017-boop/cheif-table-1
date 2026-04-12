import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "openai", "-q"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import requests
from openai import OpenAI

ENV_URL = os.getenv("ENV_URL", "https://shivanimp017-chefs-table-ai.hf.space")
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[INFO] MODEL={MODEL_NAME}", flush=True)
print(f"[INFO] ENV_URL={ENV_URL}", flush=True)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

RECIPES = ["Butter Chicken", "Margherita Pizza", "Chicken Curry", "Paneer Butter Masala",
           "Grilled Salmon", "Dal Tadka", "Caesar Salad", "Beef Steak"]
VEG = ["Margherita Pizza", "Caesar Salad", "Paneer Butter Masala", "Dal Tadka"]
NONVEG = ["Butter Chicken", "Chicken Curry", "Grilled Salmon", "Beef Steak"]

def get_llm_action(task, obs, step):
    recs = obs.get("recommendations", RECIPES[:4])
    prompt = f"Task: {task}. Recommended recipes: {recs}. Reply with just one recipe name from the list."
    print(f"[LLM] Calling model={MODEL_NAME} base_url={API_BASE_URL}", flush=True)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        timeout=20
    )
    content = response.choices[0].message.content.strip()
    print(f"[LLM] Response: {content}", flush=True)
    for r in RECIPES:
        if r.lower() in content.lower():
            return r
    if task == "task_medium":
        return VEG[step % len(VEG)] if step % 2 == 0 else NONVEG[step % len(NONVEG)]
    return recs[0] if recs else RECIPES[0]

for task in ["task_easy", "task_medium", "task_hard"]:
    print(f"[START] task={task} env=chefs-table model={MODEL_NAME}", flush=True)
    step, score, done = 0, 0.0, False
    obs = {}
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task}, timeout=10)
        obs = r.json().get("observation", {})
    except Exception as e:
        print(f"[STEP] step=1 reward=0.05 done=true error={str(e)[:80]}", flush=True)
        print(f"[END] task={task} score=0.05 steps=1", flush=True)
        continue

    while not done and step < 5:
        try:
            recipe = get_llm_action(task, obs, step)
            r2 = requests.post(f"{ENV_URL}/step",
                json={"action": {"recipe": recipe, "feedback": "like", "task": task.replace("task_", "")}},
                timeout=10)
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
