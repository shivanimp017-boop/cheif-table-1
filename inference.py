from openenv.core import Environment, Observation, Action
from rl_agent import get_recommendations, update_reward, get_q_table, RECIPE_NAMES, CATEGORIES
from typing import Optional, Any
import random


class RecipeRecommendationEnv(Environment):
    """
    Chef's Table AI - Recipe Recommendation RL Environment
    State:  User Q-table + current recommendations
    Action: {'recipe': str, 'feedback': 'like' | 'dislike'}
    Reward: +1.0 like, -0.5 dislike
    """

    def __init__(self):
        super().__init__()
        self.username = "openenv_agent"
        self.step_count = 0
        self.max_steps = 20
        self.total_reward = 0.0
        self._state = {}

    @property
    def state(self):
        return self._state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> dict:
        self.step_count = 0
        self.total_reward = 0.0
        self._state = self._build_state()
        return self._state

    def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs) -> dict:
        if isinstance(action, dict):
            recipe   = action.get('recipe', random.choice(RECIPE_NAMES))
            feedback = action.get('feedback', 'like')
        else:
            recipe   = random.choice(RECIPE_NAMES)
            feedback = 'like'

        if recipe not in RECIPE_NAMES:
            recipe = random.choice(RECIPE_NAMES)

        reward = 1.0 if feedback == 'like' else -0.5
        update_reward(self.username, recipe, reward)

        self.total_reward += reward
        self.step_count += 1
        self._state = self._build_state()

        done = self.step_count >= self.max_steps

        return {
            'state': self._state,
            'reward': reward,
            'done': done,
            'info': {
                'recipe': recipe,
                'feedback': feedback,
                'total_reward': round(self.total_reward, 3),
                'step': self.step_count,
            }
        }

    def _build_state(self) -> dict:
        q    = get_q_table(self.username)
        recs = get_recommendations(self.username, top_n=4)
        return {
            'recommendations': [r['name'] for r in recs],
            'q_values': {k: round(v, 3) for k, v in q.items()},
            'step': self.step_count,
            'total_reward': round(self.total_reward, 3),
        }


# Standard OpenEnv entry point
env = RecipeRecommendationEnv()


if __name__ == "__main__":
    print("=" * 50)
    print("Chef's Table AI - OpenEnv Test")
    print("=" * 50)

    state = env.reset()
    print("Reset OK")
    print("Recommendations:", state['recommendations'])

    for i in range(3):
        action = {
            'recipe': random.choice(RECIPE_NAMES),
            'feedback': random.choice(['like', 'dislike'])
        }
        obs = env.step(action)
        print(f"Step {i+1}: {obs['info']['recipe']} -> {obs['info']['feedback']} reward={obs['reward']}")

    print("Total reward:", env.total_reward)
    print("Test PASSED!")
