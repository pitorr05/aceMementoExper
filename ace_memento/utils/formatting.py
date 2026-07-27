import json
import re
from typing import List, Dict, Any, Optional

META_PLANNER_SYSTEM_PROMPT = """You are the META-PLANNER in a hierarchical AI system.
A user will ask a high-level question. Your task is to break down the problem into a minimal sequence of executable subtasks.
Reply ONLY in JSON.

If you need to execute subtasks to solve the question, use this schema:
{
  "plan": [
    {"id": 1, "description": "Step-by-step executable description"}
  ],
  "bullet_ids": ["calc-00001", "fin-00002"]
}

If you have enough information or results to answer the question, set "plan" to [] (empty list) and provide the final answer using this schema:
{
  "plan": [],
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "<your answer>"
}

Follow these rules for the final answer:
- The answer should be a number, or as few words as possible, or a comma-separated list.
- If it's a number, do not use commas inside the number or include units ($ or %) unless asked.
- Avoid articles and abbreviations unless specified.
- Pure JSON only. No extra commentary.

Each bullet point in the playbook has a bullet_id. Include the IDs of all bullet points in the playbook that are relevant or helpful for this question in the "bullet_ids" list.
"""

META_PLANNER_USER_TEMPLATE = """Playbook Rules & Strategies:
{playbook}

Case Memory Examples:
{cases}

Previous Task History (if any):
{history}

Current Question:
{question}

Context:
{context}

Generate your plan or final answer:
"""

EXEC_SYSTEM_PROMPT = """You are the EXECUTOR sub-agent. You receive one task description at a time from the meta-planner.
Your job is to complete the task, using available tools via function calling if needed.
Always think step by step but reply with the minimal content needed for the meta-planner.
If you must call a tool, produce the appropriate function call instead of natural language.
When done, output a concise result. Do NOT output 'FINAL ANSWER'.
"""

# XML/Markdown/JSON strip helper
def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\s\S]*}", text)
    return m.group(0) if m else text

def extract_json_from_text(text: str, key_to_find: str = None) -> Optional[Any]:
    text = text.strip()
    
    # 1. Try parsing the whole text directly
    try:
        cleaned_text = text
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```[^\n]*\n", "", cleaned_text)
            cleaned_text = re.sub(r"\n?```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            if key_to_find:
                return data.get(key_to_find)
            return data
    except Exception:
        pass

    # 2. Extract valid JSON blocks by brace counting
    stack = []
    start_idx = -1
    parsed_dicts = []
    for idx, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = idx
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    candidate = text[start_idx:idx+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            parsed_dicts.append(parsed)
                    except Exception:
                        pass

    if parsed_dicts:
        # Find the last dictionary that actually has a non-placeholder final_answer if possible, or just the last dict
        target_dict = parsed_dicts[-1]
        for d in reversed(parsed_dicts):
            fa = d.get("final_answer")
            if fa and not (isinstance(fa, str) and (fa.startswith("[") or "concise" in fa.lower() or "final answer" in fa.lower())):
                target_dict = d
                break
        
        if key_to_find:
            return target_dict.get(key_to_find)
        return target_dict

    # 3. Fallback regex extraction for fields if no JSON was parsed
    if key_to_find:
        # Try to find the key using regex
        patterns = [
            rf'"{key_to_find}"\s*:\s*"([^"]*)"',
            rf"'{key_to_find}'\s*:\s*'([^']*)'",
            rf'"{key_to_find}"\s*:\s*([^\s,}}]+)',
            rf"'{key_to_find}'\s*:\s*([^\s,}}]+)"
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # filter out placeholder values
                valid_matches = [m for m in matches if not (isinstance(m, str) and (m.startswith("[") or "concise" in m.lower() or "final answer" in m.lower()))]
                if valid_matches:
                    return valid_matches[-1]
                return matches[-1]
                
    # If no key is specified, we can construct a dummy dict using regexes for final_answer and bullet_ids
    else:
        fa = None
        for pattern in [r'"final_answer"\s*:\s*"([^"]*)"', r"'final_answer'\s*:\s*'([^']*)'", r'"final_answer"\s*:\s*([^\s,}]+)', r"'final_answer'\s*:\s*([^\s,}]+)"]:
            matches = re.findall(pattern, text)
            if matches:
                valid_matches = [m for m in matches if not (isinstance(m, str) and (m.startswith("[") or "concise" in m.lower() or "final answer" in m.lower()))]
                fa = valid_matches[-1] if valid_matches else matches[-1]
                break
                
        bids = []
        bids_matches = re.findall(r'\[?([a-z]{3,}-\d{5})\]?', text)
        if bids_matches:
            bids = list(set(bids_matches))
            
        if fa is not None:
            return {"final_answer": fa, "bullet_ids": bids}

    return None
