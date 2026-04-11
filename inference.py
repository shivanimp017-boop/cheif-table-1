from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from server.rl_agent import CATEGORIES, RECIPE_NAMES

try:
    from server.app import app as flask_app
except Exception:
    flask_app = None


TASK_LIMITS = {"easy": 10, "medium": 15, "hard": 20}
TASK_TARGETS = {"easy": 3, "medium": 4, "hard": 5.0}
POPULAR_RECIPES = [
    "Butter Chicken",
    "Margherita Pizza",
    "Chicken Curry",
    "Paneer Butter Masala",
    "Grilled Salmon",
    "Dal Tadka",
]
MEDIUM_PRIORITY = [
    "Paneer Butter Masala",
    "Butter Chicken",
    "Dal Tadka",
    "Chicken Curry",
    "Margherita Pizza",
    "Grilled Salmon",
]


class RecipeAction(Action):
    recipe: str = Field(..., description="Recipe name to interact with")
    feedback: Literal["like", "dislike"] = Field(
        ..., description="Feedback for the selected recipe"
    )
    task: Literal["easy", "medium", "hard"] | None = Field(
        default=None,
        description="Optional task override for this step",
    )


class RecipeObservation(Observation):
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended recipes for the current step",
    )
    q_values: dict[str, float] = Field(
        default_factory=dict,
        description="Current Q-values for each recipe",
    )
    step: int = Field(default=0, description="Current step count")
    total_reward: float = Field(default=0.0, description="Episode cumulative reward")
    task: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Current task identifier",
    )
    task_score: float = Field(
        default=0.0, description="Normalized task completion score in [0, 1]"
    )
    task_progress: str = Field(
        default="", description="Human-readable progress summary"
    )


class RecipeState(State):
    task: Literal["easy", "medium", "hard"] = Field(default="easy")
    total_reward: float = Field(default=0.0)
    liked_recipes: list[str] = Field(default_factory=list)
    disliked_recipes: list[str] = Field(default_factory=list)
    seen_recipes: list[str] = Field(default_factory=list)
    liked_veg: int = Field(default=0, ge=0)
    liked_nonveg: int = Field(default=0, ge=0)
    q_values: dict[str, float] = Field(default_factory=dict)


class RecipeRecommendationEnv(
    Environment[RecipeAction, RecipeObservation, RecipeState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = self._fresh_state()

    @property
    def state(self) -> RecipeState:
        return self._state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> RecipeObservation:
        normalized_task = self._normalize_task(task_id or task)
        self._state = self._fresh_state(task=normalized_task, episode_id=episode_id)
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: RecipeAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> RecipeObservation:
        del timeout_s

        if action.task:
            self._state.task = self._normalize_task(action.task)

        recipe = action.recipe.strip()
        if recipe not in RECIPE_NAMES:
            self._state.step_count += 1
            return self._build_observation(
                reward=-1.0,
                done=self._is_done(),
                progress_override=f"Unknown recipe '{recipe}'. Choose one of the listed recipes.",
            )

        if recipe not in self._state.seen_recipes:
            self._state.seen_recipes.append(recipe)

        was_liked_before = recipe in self._state.liked_recipes
        reward = 1.0 if action.feedback == "like" else -0.5
        if action.feedback == "like" and not was_liked_before:
            reward += 0.3
            self._state.liked_recipes.append(recipe)
            if CATEGORIES.get(recipe) == "veg":
                self._state.liked_veg += 1
            else:
                self._state.liked_nonveg += 1
        elif action.feedback == "dislike" and recipe not in self._state.disliked_recipes:
            self._state.disliked_recipes.append(recipe)

        self._update_q_value(recipe, reward)
        self._state.total_reward = round(self._state.total_reward + reward, 4)
        self._state.step_count += 1

        done = self._is_done()
        return self._build_observation(reward=reward, done=done)

    def _fresh_state(
        self,
        task: Literal["easy", "medium", "hard"] = "easy",
        episode_id: str | None = None,
    ) -> RecipeState:
        return RecipeState(
            episode_id=episode_id,
            task=task,
            total_reward=0.0,
            liked_recipes=[],
            disliked_recipes=[],
            seen_recipes=[],
            liked_veg=0,
            liked_nonveg=0,
            q_values={recipe: 0.0 for recipe in RECIPE_NAMES},
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        progress_override: str | None = None,
    ) -> RecipeObservation:
        return RecipeObservation(
            done=done,
            reward=reward,
            recommendations=self._recommendations(),
            q_values=dict(self._state.q_values),
            step=self._state.step_count,
            total_reward=self._state.total_reward,
            task=self._state.task,
            task_score=self._task_score(),
            task_progress=progress_override or self._task_progress(),
            metadata={
                "max_steps": TASK_LIMITS[self._state.task],
                "remaining_steps": max(
                    0, TASK_LIMITS[self._state.task] - self._state.step_count
                ),
            },
        )

    def _recommendations(self) -> list[str]:
        prioritized = self._priority_list()
        scored: list[tuple[float, str]] = []

        for rank, recipe in enumerate(prioritized):
            q_value = self._state.q_values.get(recipe, 0.0)
            unseen_bonus = 1.5 if recipe not in self._state.seen_recipes else 0.0
            novelty_bonus = 0.6 if recipe not in self._state.liked_recipes else -0.2
            priority_bonus = max(0.0, 1.0 - (rank * 0.03))
            scored.append((q_value + unseen_bonus + novelty_bonus + priority_bonus, recipe))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [recipe for _, recipe in scored[:4]]

    def _priority_list(self) -> list[str]:
        if self._state.task == "medium":
            preferred = MEDIUM_PRIORITY
        else:
            preferred = POPULAR_RECIPES

        remaining = [recipe for recipe in RECIPE_NAMES if recipe not in preferred]
        return preferred + remaining

    def _update_q_value(self, recipe: str, reward: float) -> None:
        alpha = 0.3
        gamma = 0.9
        current_value = self._state.q_values.get(recipe, 0.0)
        max_future_value = max(self._state.q_values.values(), default=0.0)
        updated_value = current_value + alpha * (
            reward + (gamma * max_future_value) - current_value
        )
        self._state.q_values[recipe] = round(updated_value, 4)

    def _task_score(self) -> float:
        if self._state.task == "easy":
            return round(min(1.0, len(set(self._state.liked_recipes)) / 3.0), 3)
        if self._state.task == "medium":
            progress = min(self._state.liked_veg, 2) + min(self._state.liked_nonveg, 2)
            return round(min(1.0, progress / 4.0), 3)
        return round(min(1.0, self._state.total_reward / TASK_TARGETS["hard"]), 3)

    def _task_progress(self) -> str:
        if self._state.task == "easy":
            return f"{len(set(self._state.liked_recipes))}/3 unique recipe likes"
        if self._state.task == "medium":
            return (
                f"veg={self._state.liked_veg}/2 | "
                f"non-veg={self._state.liked_nonveg}/2"
            )
        return f"total_reward={self._state.total_reward:.1f}/5.0"

    def _is_done(self) -> bool:
        limit_reached = self._state.step_count >= TASK_LIMITS[self._state.task]
        if self._state.task == "easy":
            return limit_reached or len(set(self._state.liked_recipes)) >= 3
        if self._state.task == "medium":
            return (
                limit_reached
                or self._state.liked_veg >= 2
                and self._state.liked_nonveg >= 2
            )
        return limit_reached or self._state.total_reward >= TASK_TARGETS["hard"]

    @staticmethod
    def _normalize_task(task: str | None) -> Literal["easy", "medium", "hard"]:
        value = (task or "easy").strip().lower()
        if value.startswith("task_"):
            value = value[5:]
        if value not in TASK_LIMITS:
            return "easy"
        return value  # type: ignore[return-value]


app = create_app(
    RecipeRecommendationEnv,
    RecipeAction,
    RecipeObservation,
    env_name="chefs-table-ai",
    max_concurrent_envs=4,
)

if flask_app is not None:
    app.mount("/site", WSGIMiddleware(flask_app))


@app.get("/", include_in_schema=False)
async def root(_: Request) -> RedirectResponse:
    if flask_app is not None:
        return RedirectResponse(url="/site/")
    return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port or int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
