"""
Agentic AI Supervisor
Monitors the pipeline and provides hooks for future user-intent-driven modifications.

STATUS: Not yet activated in production.
  - interpret_user_intent() is defined but never called by the orchestrator.
  - _rule_based_intent() contains placeholder keyword rules (brick/concrete/steel).
  - When ready to activate: wire interpret_user_intent() into the orchestrator after
    Stage 5 (semantic analysis) and replace _rule_based_intent() with an LLM call.

NOTE: Local Qwen backend is not yet activated.  When GPU hardware is available,
swap _call_llm() for a real Qwen2.5-VL inference call and call it from here.
"""

from loguru import logger


class SystemSupervisor:
    """Autonomous agent that supervises the generation process."""

    async def interpret_user_intent(self, user_prompt: str, current_recipe: dict) -> dict:
        """
        Modify the Revit recipe based on a natural language user prompt.
        Currently implements simple keyword rules; replace _call_llm() with a
        real model call when GPU is available.
        """
        logger.info(f"Supervisor interpreting: '{user_prompt}'")
        return self._rule_based_intent(user_prompt)

    async def monitor_pipeline(self, job_id: str, status: str, error: str = None):
        """Log pipeline health; add auto-recovery logic here when needed."""
        if status == "failed":
            logger.error(f"Supervisor detected failure in job {job_id}: {error}")

    # ── Intent rules (placeholder until LLM is wired in) ──────────────────────

    def _rule_based_intent(self, prompt: str) -> dict:
        p = prompt.lower()
        if "brick" in p:
            return {"action": "modify_all", "target_command": "Wall.Create",
                    "modifications": {"WallType": "Brick - Common"}}
        if "concrete" in p:
            return {"action": "modify_all", "target_command": "Wall.Create",
                    "modifications": {"WallType": "Concrete - 200mm"}}
        if "steel" in p:
            return {"action": "modify_all", "target_command": "Column.Create",
                    "modifications": {"ColumnType": "Steel - UC 203x203x46"}}
        return {"action": "none", "reason": "No actionable intent found"}
