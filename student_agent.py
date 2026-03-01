import json
import re

class AssemblyAgent:
    def __init__(self):
        self._re_init_goal = re.compile(
            r"As initial conditions I have that,\s*(.*?)\.\s*My goal is to have that\s*(.*?)\.\s*My plan is as follows:",
            re.S | re.I
        )

        self.system_blocks = (
            "You are a deterministic planner. Temperature=0, greedy decoding.\n"
            "Return ONLY a JSON array of action strings.\n"
            "Find the SHORTEST plan. If multiple shortest plans exist, break ties by BFS expansion order.\n\n"
            "DOMAIN: BLOCKS\n"
            "Canonical action syntax (must match exactly):\n"
            '- "(engage_payload X)"\n'
            '- "(release_payload X)"\n'
            '- "(unmount_node X Y)"\n'
            '- "(mount_node X Y)"\n'
            "Use lowercase action names. X,Y are block names (e.g., red, blue, orange, yellow).\n"
            "Each action string must have exactly one space between tokens and be wrapped in parentheses.\n\n"
            "State concepts:\n"
            "- hand_empty, holding(X), on(X,table) or on(X,Y), clear(X).\n\n"
            "Action rules:\n"
            "1) engage_payload X (pick up): pre hand_empty & on(X,table) & clear(X). eff holding(X) & not hand_empty.\n"
            "2) release_payload X (put down): pre holding(X). eff on(X,table) & hand_empty.\n"
            "3) unmount_node X Y (unstack): pre hand_empty & on(X,Y) & clear(X). eff holding(X) & clear(Y).\n"
            "4) mount_node X Y (stack): pre holding(X) & clear(Y). eff on(X,Y) & hand_empty & not clear(Y).\n\n"
            "BFS tie-break action order (very important):\n"
            "- If hand_empty: list all unmount_node actions first (sort by X then Y), then engage_payload actions (sort by X).\n"
            "- If holding(X): list all mount_node actions first (sort by Y), then release_payload.\n\n"
            "Output format example:\n"
            '["(unmount_node red blue)", "(release_payload red)"]\n'
            "No extra text."
        )

        self.system_objects = (
            "You are a deterministic planner. Temperature=0, greedy decoding.\n"
            "Return ONLY a JSON array of action strings.\n"
            "Find the SHORTEST plan. If multiple shortest plans exist, break ties by BFS expansion order.\n\n"
            "DOMAIN: OBJECTS\n"
            "Canonical action syntax (must match exactly):\n"
            '- "(attack x)"\n'
            '- "(feast x y)"\n'
            '- "(succumb x)"\n'
            '- "(overcome x y)"\n'
            "Use lowercase action names. x,y are object ids (e.g., a,b,c).\n"
            "Each action string must have exactly one space between tokens and be wrapped in parentheses.\n\n"
            "Facts:\n"
            "- harmony (boolean)\n"
            "- province(x), planet(x), pain(x)\n"
            "- craves(x,y)\n\n"
            "Action rules:\n"
            "1) attack(x): pre harmony & province(x) & planet(x). eff pain(x). del harmony, province(x), planet(x).\n"
            "2) succumb(x): pre pain(x). eff harmony, province(x), planet(x). del pain(x).\n"
            "3) feast(x,y): pre harmony & province(x) & craves(x,y). eff pain(x), province(y). del harmony, province(x), craves(x,y).\n"
            "4) overcome(x,y): pre pain(x) & province(y). eff harmony, province(x), craves(x,y). del province(y), pain(x).\n\n"
            "BFS tie-break action order (very important):\n"
            "- If any pain(.) is true: consider succumb(x) first (sort by x), then overcome(x,y) (sort by x then y).\n"
            "- Else (no pain) and harmony is true: consider feast(x,y) first (sort by x then y), then attack(x) (sort by x).\n\n"
            "Output format example:\n"
            '["(feast a b)", "(succumb a)"]\n'
            "No extra text."
        )

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        stmt = self._second_statement(scenario_context)
        init_text, goal_text = self._extract_init_goal(stmt)

        is_blocks = ("set of blocks" in scenario_context.lower())
        system = self.system_blocks if is_blocks else self.system_objects

        user_prompt = (
            "INITIAL:\n" + init_text + "\n\n"
            "GOAL:\n" + goal_text + "\n\n"
            "Return ONLY the JSON array."
        )

        out = llm_engine_func(
            prompt=user_prompt,
            system=system,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=220
        )

        plan = self._parse_json_array(out, llm_engine_func, system)
        return plan

    def _second_statement(self, context: str) -> str:
        parts = context.split("[STATEMENT]")
        return parts[-1] if len(parts) > 1 else context

    def _extract_init_goal(self, stmt: str):
        m = self._re_init_goal.search(stmt)
        if not m:
            raise ValueError("No se pudo parsear initial/goal del segundo [STATEMENT].")
        return m.group(1).strip(), m.group(2).strip()

    def _parse_json_array(self, text: str, llm_engine_func, system: str):
        s = text.strip()
        l = s.find("[")
        r = s.rfind("]")
        if l != -1 and r != -1 and r > l:
            s = s[l:r+1]
        try:
            arr = json.loads(s)
            if not isinstance(arr, list):
                raise ValueError("JSON no es lista")
            arr2 = []
            for a in arr:
                if not isinstance(a, str):
                    continue
                a = a.strip()
                if a.startswith("(") and a.endswith(")"):
                    arr2.append(a)
            return arr2
        except Exception:
            fix_prompt = (
                "Fix the following into a VALID JSON array of action strings ONLY.\n"
                "Do not add any explanations.\n\n"
                "BROKEN_OUTPUT:\n"
                + text.strip()
            )
            out2 = llm_engine_func(
                prompt=fix_prompt,
                system=system,
                temperature=0.0,
                do_sample=False,
                max_new_tokens=220
            )
            s2 = out2.strip()
            l2 = s2.find("[")
            r2 = s2.rfind("]")
            if l2 != -1 and r2 != -1 and r2 > l2:
                s2 = s2[l2:r2+1]
            arr = json.loads(s2)
            if not isinstance(arr, list):
                return []
            return [x.strip() for x in arr if isinstance(x, str) and x.strip().startswith("(") and x.strip().endswith(")")]
 