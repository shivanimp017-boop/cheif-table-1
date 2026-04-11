"""
Baseline inference script for Chef's Table AI RL Environment
Runs all 3 tasks and prints reproducible scores.
"""
from inference import RecipeRecommendationEnv, RecipeAction

def run_task(task: str):
    env = RecipeRecommendationEnv()
    obs = env.reset(task=task)
    print(f"\n--- Task: {task.upper()} ---")
    print(f"Goal: {obs.task_progress}")

    from server.rl_agent import RECIPE_NAMES, CATEGORIES
    import random

    for i in range(20):
        # Simple baseline: like recommended recipes
        recipe = obs.recommendations[0] if obs.recommendations else random.choice(RECIPE_NAMES)

        # Medium task: alternate veg/nonveg
        if task == "medium":
            veg = [r for r in RECIPE_NAMES if CATEGORIES.get(r) == 'veg']
            nonveg = [r for r in RECIPE_NAMES if CATEGORIES.get(r) == 'nonveg']
            recipe = veg[i % len(veg)] if i % 2 == 0 else nonveg[i % len(nonveg)]

        action = RecipeAction(recipe=recipe, feedback="like", task=task)
        obs = env.step(action)

        print(f"  Step {obs.step}: liked '{recipe}' | reward={obs.reward} | score={obs.task_score} | {obs.task_progress}")

        if obs.done:
            break

    print(f"Final Score: {obs.task_score:.3f} / 1.0")
    return obs.task_score

if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task)

    print("\n========== BASELINE SCORES ==========")
    for task, score in scores.items():
        print(f"  {task.upper()}: {score:.3f} / 1.0")
    print(f"  AVERAGE: {sum(scores.values())/len(scores):.3f} / 1.0")
    print("=====================================")
