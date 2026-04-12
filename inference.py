import os
import sys
import random
import socket
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core import Environment, Action, Observation, create_fastapi_app
from rl_agent import RECIPE_NAMES, CATEGORIES
import uvicorn


class RecipeAction(Action):
    recipe: str = ""
    feedback: str = "like"
    task: str = "easy"


class RecipeObservation(Observation):
    recommendations: list
    q_values: dict
    step: int
    total_reward: float
    reward: float = 0.0
    done: bool = False
    task: str = "easy"
    task_score: float = 0.0
    task_progress: str = ""


class RecipeRecommendationEnv(Environment):
    """Chef's Table AI — Recipe Recommendation RL Environment"""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._reset_state("easy")

    def _reset_state(self, task="easy"):
        self.step_count = 0
        self.total_reward = 0.0
        self.task = task
        self.liked_recipes = []
        self.liked_veg = 0
        self.liked_nonveg = 0
        self._q = {r: 0.0 for r in RECIPE_NAMES}

    @property
    def state(self) -> RecipeObservation:
        return self._build_obs(0.0, False)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RecipeObservation:
        task = kwargs.get("task", "easy")
        if isinstance(task, str) and task.startswith("task_"):
            task = task[5:]
        self._reset_state(task if task in ("easy", "medium", "hard") else "easy")
        return self._build_obs(0.0, False)

    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RecipeObservation:
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(self, action: RecipeAction, timeout_s: Optional[float] = None, **kwargs) -> RecipeObservation:
        recipe = action.recipe if action.recipe in RECIPE_NAMES else random.choice(RECIPE_NAMES)
        feedback = action.feedback if action.feedback in ('like', 'dislike') else 'like'
        task = action.task
        if isinstance(task, str) and task.startswith("task_"):
            task = task[5:]
        if task in ("easy", "medium", "hard"):
            self.task = task

        base_reward = 1.0 if feedback == 'like' else -0.5
        partial = 0.0
        if feedback == 'like' and recipe not in self.liked_recipes:
            self.liked_recipes.append(recipe)
            partial = 0.3
            if CATEGORIES.get(recipe) == 'veg':
                self.liked_veg += 1
            else:
                self.liked_nonveg += 1

        reward = base_reward + partial
        alpha, gamma = 0.3, 0.9
        max_q = max(self._q.values())
        self._q[recipe] = self._q[recipe] + alpha * (base_reward + gamma * max_q - self._q[recipe])
        self.total_reward = round(self.total_reward + reward, 4)
        self.step_count += 1

        return self._build_obs(reward, self._check_done())

    async def step_async(self, action: RecipeAction, timeout_s: Optional[float] = None, **kwargs) -> RecipeObservation:
        return self.step(action, timeout_s=timeout_s, **kwargs)

    def _check_done(self) -> bool:
        if self.step_count >= 10:
            return True
        if self.task == "easy" and len(self.liked_recipes) >= 3:
            return True
        if self.task == "medium" and self.liked_veg >= 2 and self.liked_nonveg >= 2:
            return True
        if self.task == "hard" and self.total_reward >= 5.0:
            return True
        return False

    def _grade_task(self) -> float:
        if self.task == "easy":
            return min(len(self.liked_recipes) / 3.0, 1.0)
        elif self.task == "medium":
            return (min(self.liked_veg / 2.0, 1.0) + min(self.liked_nonveg / 2.0, 1.0)) / 2.0
        return min(self.total_reward / 5.0, 1.0)

    def _task_progress(self) -> str:
        if self.task == "easy":
            return f"Liked {len(self.liked_recipes)}/3 unique recipes"
        elif self.task == "medium":
            return f"Veg: {self.liked_veg}/2, Non-veg: {self.liked_nonveg}/2"
        return f"Total reward: {round(self.total_reward, 2)}/5.0"

    def _build_obs(self, reward: float, done: bool) -> RecipeObservation:
        top = sorted(self._q.items(), key=lambda x: x[1], reverse=True)
        recs = [r for r, _ in top[:4]]
        return RecipeObservation(
            recommendations=recs,
            q_values={k: round(v, 3) for k, v in self._q.items()},
            step=self.step_count,
            total_reward=self.total_reward,
            reward=round(reward, 3),
            done=done,
            task=self.task,
            task_score=round(self._grade_task(), 3),
            task_progress=self._task_progress(),
        )


app = create_fastapi_app(
    env=RecipeRecommendationEnv,
    action_cls=RecipeAction,
    observation_cls=RecipeObservation,
)


def get_free_port():
    preferred = int(os.environ.get("PORT", 7860))
    for port in [preferred, 8000, 8080]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return preferred


def main():
    port = get_free_port()
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
