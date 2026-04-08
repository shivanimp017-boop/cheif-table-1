import subprocess
import sys
import os
import random
import socket

from openenv.core import Environment, Action, Observation, create_fastapi_app
from rl_agent import get_recommendations, update_reward, get_q_table, RECIPE_NAMES, CATEGORIES
from typing import Optional
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
    """
    Chef's Table AI — Recipe Recommendation RL Environment
    3 tasks: easy (like popular), medium (balanced veg/nonveg), hard (maximize score)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.username = "openenv_agent"
        self.step_count = 0
        self.max_steps = 20
        self.total_reward = 0.0
        self.task = "easy"
        self.liked_recipes = []
        self.liked_veg = 0
        self.liked_nonveg = 0
        self._current_state = self._build_obs(0.0, False)

    @property
    def state(self) -> RecipeObservation:
        return self._current_state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RecipeObservation:
        self.step_count = 0
        self.total_reward = 0.0
        self.liked_recipes = []
        self.liked_veg = 0
        self.liked_nonveg = 0
        self.task = kwargs.get("task", "easy")
        self._current_state = self._build_obs(0.0, False)
        return self._current_state

    def step(self, action: RecipeAction, timeout_s: Optional[float] = None, **kwargs) -> RecipeObservation:
        recipe = action.recipe if action.recipe in RECIPE_NAMES else random.choice(RECIPE_NAMES)
        feedback = action.feedback if action.feedback in ('like', 'dislike') else 'like'
        self.task = action.task if action.task in ('easy', 'medium', 'hard') else self.task

        # Base reward
        base_reward = 1.0 if feedback == 'like' else -0.5

        # Partial progress reward based on task
        partial = 0.0
        if feedback == 'like':
            if recipe not in self.liked_recipes:
                self.liked_recipes.append(recipe)
                partial = 0.3  # partial progress for new recipe liked
            cat = CATEGORIES.get(recipe, '')
            if cat == 'veg':
                self.liked_veg += 1
            else:
                self.liked_nonveg += 1

        reward = base_reward + partial
        update_reward(self.username, recipe, base_reward)

        self.total_reward += reward
        self.step_count += 1

        done = self._check_done()
        self._current_state = self._build_obs(reward, done)
        return self._current_state

    def _check_done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if self.task == "easy" and len(self.liked_recipes) >= 3:
            return True
        if self.task == "medium" and self.liked_veg >= 2 and self.liked_nonveg >= 2:
            return True
        if self.task == "hard" and self.total_reward >= 5.0:
            return True
        return False

    def _grade_task(self) -> float:
        """Returns score 0.0-1.0 based on task completion"""
        if self.task == "easy":
            return min(len(self.liked_recipes) / 3.0, 1.0)
        elif self.task == "medium":
            veg_score = min(self.liked_veg / 2.0, 1.0)
            nonveg_score = min(self.liked_nonveg / 2.0, 1.0)
            return (veg_score + nonveg_score) / 2.0
        elif self.task == "hard":
            return min(self.total_reward / 5.0, 1.0)
        return 0.0

    def _task_progress(self) -> str:
        if self.task == "easy":
            return f"Liked {len(self.liked_recipes)}/3 unique recipes"
        elif self.task == "medium":
            return f"Veg: {self.liked_veg}/2, Non-veg: {self.liked_nonveg}/2"
        elif self.task == "hard":
            return f"Total reward: {round(self.total_reward, 2)}/5.0"
        return ""

    def _build_obs(self, reward: float, done: bool) -> RecipeObservation:
        q = get_q_table(self.username)
        recs = get_recommendations(self.username, top_n=4)
        return RecipeObservation(
            recommendations=[r['name'] for r in recs],
            q_values={k: round(v, 3) for k, v in q.items()},
            step=self.step_count,
            total_reward=round(self.total_reward, 3),
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
    import socket
    preferred = int(os.environ.get("PORT", 7860))
    for port in [preferred, 8000, 8080, 8888, 9000]:
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
