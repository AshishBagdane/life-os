import json
import re

class JSONParser:
    """
    Single Responsibility: Safely extracts and parses JSON from unpredictable LLM string outputs.
    """
    @staticmethod
    def parse_llm_output(raw_text: str, agent_name: str = "Unknown Agent") -> dict:
        # 1. Try clean parsing (Fast string stripping)
        clean_text = raw_text.strip("`").removeprefix("json").strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # 2. Fallback to Regex for chatty text
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not match:
                print(f"🚨 Parsing failed for {agent_name}. Raw:\n{raw_text}")
                raise ValueError(f"No JSON block found in response from {agent_name}.")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"🚨 Regex failed for {agent_name}. Extracted:\n{match.group(0)}")
                raise e