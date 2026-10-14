"""LLMに送るシステムプロンプトのテンプレートと組み立てを管理するモジュール.

共通テンプレートにTaskProfileの4要素を差し込み、
タスク種別ごとのシステムプロンプトをbuild_system_prompt()で生成する。
"""
from src.agent.prompt.task_profile.task_profile import TaskProfile
from src.models.sandbox_base import SandboxBase


# 全タスク共通のシステムプロンプトの骨格。
# {}で囲まれた部分に、build_system_prompt()でTaskProfileの4要素を差し込む。
COMMON_TEMPLATE = """You are Agent Smith, an autonomous coding agent.
Loop: Thought (one line) -> ONE Python code block -> read the Observation.

## Format
- One brief Thought line (20 words max), then exactly ONE block fenced as ```python ... ```,
  then <end_code>.
- NEVER derive formulas or trace arithmetic in prose. Your turn is cut off at a hard output
  limit, and a reply that never closes a ```python block runs nothing and burns the budget.
  Write the candidate in code and let the Observation correct you.
- Only what you print() (plus errors and tracebacks) comes back in the Observation.
- The namespace persists between turns, but your FINAL answer must be SELF-CONTAINED.
- Limits on iterations, tokens and time are strict: keep turns short, never print large output.

## Environment and tools
{tools_manual}

## Submitting
Call final_answer(...) when you are confident. {final_answer_specification}

## Example (format only - NOT the real task)
{few_shot_example}

## This task
{task_instructions}
"""


def build_system_prompt(profile: TaskProfile, sandbox: SandboxBase) -> str:
    """TaskProfileとSandboxから、タスク種別ごとのシステムプロンプトを組み立てる.

    Args:
        profile (TaskProfile): タスク種別ごとのプロンプト差分を提供するTaskProfileの具象クラス。
        sandbox (SandboxBase): 接続中のサンドボックスのインスタンス。

    Returns:
        str: 組み立てられたシステムプロンプト
    """
    return COMMON_TEMPLATE.format(
        tools_manual=profile.manual(sandbox),
        final_answer_specification=profile.final_answer_specification(),
        few_shot_example=profile.few_shot_example(),
        task_instructions=profile.task_instructions()
    )
