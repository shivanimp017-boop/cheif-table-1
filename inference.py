import subprocess, sys, os
try:
    from openenv.core import Environment, Action, Observation, create_fastapi_app
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openenv-core", "fastapi", "uvicorn", "pydantic", "-q"])
    from openenv.core import Environment, Action, Observation, create_fastapi_app
from rl_agent import get_recommendations, update_reward, get_q_table, RECIPE_NAMES, CATEGORIES
from pydantic import BaseModel
from typing import Optional, Any
import random
import uvicorn


# ── Action & Observation models ──────────────────────────────────────────────

class RecipeAction(Action):
    recipe: str = ""
    feedback: str = "like"   # "like" or "dislike"


class RecipeObservation(Observation):
    recommendations: list
    q_values: dict
    step: int
    total_reward: float
    reward: float = 0.0
    done: bool = False


# ── Environment ───────────────────────────────────────────────────────────────

class RecipeRecommendationEnv(Environment):
    """
    Chef's Table AI — Recipe Recommendation RL Environment
    Q-Learning agent learns user food preferences over time.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.username     = "openenv_agent"
        self.step_count   = 0
        self.max_steps    = 20
        self.total_reward = 0.0
        self._current_state = self._build_obs(0.0, False)

    @property
    def state(self) -> RecipeObservation:
        return self._current_state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RecipeObservation:
        self.step_count   = 0
        self.total_reward = 0.0
        self._current_state = self._build_obs(0.0, False)
        return self._current_state

    def step(self, action: RecipeAction, timeout_s: Optional[float] = None, **kwargs) -> RecipeObservation:
        recipe   = action.recipe   if action.recipe   in RECIPE_NAMES else random.choice(RECIPE_NAMES)
        feedback = action.feedback if action.feedback in ('like', 'dislike') else 'like'

        reward = 1.0 if feedback == 'like' else -0.5
        update_reward(self.username, recipe, reward)

        self.total_reward += reward
        self.step_count   += 1

        done = self.step_count >= self.max_steps
        self._current_state = self._build_obs(reward, done)
        return self._current_state

    def _build_obs(self, reward: float, done: bool) -> RecipeObservation:
        q    = get_q_table(self.username)
        recs = get_recommendations(self.username, top_n=4)
        return RecipeObservation(
            recommendations = [r['name'] for r in recs],
            q_values        = {k: round(v, 3) for k, v in q.items()},
            step            = self.step_count,
            total_reward    = round(self.total_reward, 3),
            reward          = reward,
            done            = done,
        )


# ── FastAPI app (OpenEnv standard) ────────────────────────────────────────────

app = create_fastapi_app(
    env              = RecipeRecommendationEnv,
    action_cls       = RecipeAction,
    observation_cls  = RecipeObservation,
)


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
