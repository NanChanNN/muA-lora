import re
from typing import Optional, List


FORMAT_PATTERN = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")


def _gsm8k_extract_final_answer_from_completion(text: str) -> Optional[str]:
    """Extract a numeric final answer from a model completion.

    Default format: "<answer>42</answer>"
    We are also tolerant with more math-answer formats:
      - "#### 42" (GSM8K convention)
      - "\\boxed{42}" (common LaTeX convention)
    """
    input_text = str(text).strip()

    # XML-like answer tag
    m = re.search(r"<answer>\s*([-+]?\d[\d,]*)\s*</answer>", input_text)
    if m:
        return m.group(1).replace(",", "").strip()

    # explicit GSM8K delimiter
    m = re.search(r"####\s*([-+]?\d[\d,]*)", input_text)
    if m:
        return m.group(1).replace(",", "").strip()

    # LaTeX boxed
    m = re.search(r"\\boxed\{\s*([-+]?\d[\d,]*)\s*\}", input_text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Fallback: last integer-like token in the text
    nums = re.findall(r"[-+]?\d[\d,]*", input_text)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip()


def gsm8k_accuracy_reward_func(
    completions: List[str],
    solution: List[str],
    reward_point: float,
    **kwargs,
) -> List[float]:
    """Binary correctness reward for GSM8K.
    """
    rewards: List[float] = []
    for content, gt in zip(completions, solution):
        pred = _gsm8k_extract_final_answer_from_completion(content)
        gt_norm = str(gt).replace(",", "").strip()

        correct = (pred is not None) and (gt_norm is not None) and (pred == gt_norm)
        r = reward_point if correct else 0.0

        rewards.append(float(r))
    return rewards


def general_format_reward_func(
        completions: List[str],
        reward_point: float,
        **kwargs,
):
    """General formatting reward."""
    rewards: List[float] = []
    for content in completions:
        if FORMAT_PATTERN.match(content):
            rewards.append(reward_point)
        else:
            rewards.append(0.0)
    return rewards


def _extract_latex_boxed(text: str) -> Optional[str]:
    """Extract content from the first \\boxed{...} occurrence using brace matching."""
    idx = text.find(r"\boxed")
    if idx == -1:
        return None

    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None

    virtual_stack = 0
    for j in range(brace_start, len(text)):
        if text[j] == "{":
            virtual_stack += 1
        elif text[j] == "}":
            virtual_stack -= 1
            if virtual_stack == 0:
                return text[brace_start + 1:j]
    return None


def _normalize_math_answer(ans: Optional[str]) -> Optional[str]:
    """Canonicalize math answers for stricter but less brittle string matching."""
    if ans is None:
        return None
    s = str(ans).strip()

    # Strip <answer> wrappers if they appear in ground truth or completion by accident
    s = re.sub(r"^\s*<answer>\s*", "", s)
    s = re.sub(r"\s*</answer>\s*$", "", s)

    # Strip $...$ (common LaTeX inline math)
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # Strip \( ... \) and \[ ... \]
    s = re.sub(r"^\s*\\\(\s*", "", s)
    s = re.sub(r"\s*\\\)\s*$", "", s)
    s = re.sub(r"^\s*\\\[\s*", "", s)
    s = re.sub(r"\s*\\\]\s*$", "", s)

    # Remove \left and \right (formatting-only)
    s = s.replace(r"\left", "").replace(r"\right", "")

    # If the whole answer is boxed, unbox it
    if r"\boxed" in s:
        boxed = _extract_latex_boxed(s)
        if boxed is not None:
            s = boxed.strip()

    # Collapse common text wrappers
    s = re.sub(r"\\text\{\s*([^}]*)\s*\}", r"\1", s)
    s = re.sub(r"\\mathrm\{\s*([^}]*)\s*\}", r"\1", s)

    # Remove LaTeX spacing commands
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "").replace(r"\!", "").replace(r"\\ ", "")

    # Normalize whitespace: remove all whitespace to make "p - q" match "p-q"
    s = re.sub(r"\s+", "", s)

    return s if s else None


def _math_extract_answer_from_completion(text: str) -> Optional[str]:
    """Extract a (possibly LaTeX) final answer string from a completion."""
    input_text = str(text).strip()

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", input_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"####\s*(.+)", input_text)
    if m:
        return m.group(1).splitlines()[0].strip()

    boxed = _extract_latex_boxed(input_text)
    if boxed is not None:
        return boxed.strip()

    m = re.findall(r"\$\s*(.*?)\s*\$", input_text)
    if m:
        return m[-1].strip()

    # 4) last non-empty line fallback
    lines = [ln.strip() for ln in input_text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def math_accuracy_reward_func(
        completions: List[str],
        solution: List[str],
        reward_point: float,
        **kwargs,
) -> List[float]:
    """Binary correctness reward for MATH500.
    """
    rewards: List[float] = []
    for content, gt in zip(completions, solution):
        pred_raw = _math_extract_answer_from_completion(content)

        pred_norm = _normalize_math_answer(pred_raw)
        gt_norm = _normalize_math_answer(gt)

        correct = (pred_norm is not None) and (gt_norm is not None) and (pred_norm == gt_norm)
        r = reward_point if correct else 0.0

        rewards.append(float(r))
    return rewards


