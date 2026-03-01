import re
from collections import deque

class AssemblyAgent:


    def __init__(self):
        self._did_llm_warmup = False
        self.last_complexity_level = None

    def solve(self, scenario_context: str, llm_engine_func) -> list:
       
        if (not self._did_llm_warmup) and llm_engine_func is not None:
            try:
                _ = llm_engine_func(
                    prompt="Reply with OK.",
                    system="You are a deterministic assistant. Reply only with OK.",
                    temperature=0.0,
                    do_sample=False,
                    max_new_tokens=2,
                )
            except Exception:
               
                pass
            self._did_llm_warmup = True

        stmt = self._extract_second_statement(scenario_context)
        init_text, goal_text = self._parse_initial_goal(stmt)

        if "set of blocks" in scenario_context.lower():
            plan = self._solve_blocks(init_text, goal_text)
        else:
            plan = self._solve_objects(init_text, goal_text)

        self.last_complexity_level = len(plan)
        return plan

    # -------------------------
    # Parsing helpers
    # -------------------------
    def _extract_second_statement(self, context: str) -> str:
        parts = context.split("[STATEMENT]")
        if len(parts) < 2:
            raise ValueError("No [STATEMENT] blocks found.")
        return parts[-1]

    def _parse_initial_goal(self, statement_block: str):
        m = re.search(
            r"As initial conditions I have that,\s*(.*?)\.\s*My goal is to have that\s*(.*?)\.\s*My plan is as follows:",
            statement_block,
            re.S | re.I,
        )
        if not m:
            m = re.search(
                r"As initial conditions I have that,\s*(.*?)\nMy goal is to have that\s*(.*?)\n\nMy plan is as follows:",
                statement_block,
                re.S | re.I,
            )
        if not m:
            raise ValueError("Could not parse initial conditions / goal.")
        return m.group(1).strip(), m.group(2).strip()

    def _split_facts(self, text: str):
       
        t = text.replace(" and ", ", ")
        facts = [f.strip() for f in t.split(",") if f.strip()]
        return facts

    # -------------------------
    # Domain 1: Blocks world
    # -------------------------
    def _parse_blocks_init(self, init_text: str):
        facts = self._split_facts(init_text)
        on = {}
        blocks = set()
        holding = None 

        for f in facts:
            f = f.strip().lower()
            if f == "the hand is empty":
                holding = None
                continue

            m = re.match(r"the (\w+) block is on top of the (\w+) block", f)
            if m:
                a, b = m.group(1), m.group(2)
                on[a] = b
                blocks.update([a, b])
                continue

            m = re.match(r"the (\w+) block is on the table", f)
            if m:
                a = m.group(1)
                on[a] = "table"
                blocks.add(a)
                continue

            m = re.match(r"the (\w+) block is unobstructed", f)
            if m:
                blocks.add(m.group(1))
                continue

        for b in blocks:
            on.setdefault(b, "table")

        return holding, on

    def _parse_blocks_goal(self, goal_text: str):
        facts = self._split_facts(goal_text)
        goals = []
        for f in facts:
            f = f.strip().lower()
            m = re.match(r"the (\w+) block is on top of the (\w+) block", f)
            if m:
                goals.append(("on", m.group(1), m.group(2)))
                continue
            m = re.match(r"the (\w+) block is on the table", f)
            if m:
                goals.append(("table", m.group(1), None))
                continue
        return goals

    def _solve_blocks(self, init_text: str, goal_text: str):
        holding, on_map = self._parse_blocks_init(init_text)
        goals = self._parse_blocks_goal(goal_text)

        blocks = sorted(on_map.keys())
        idx = {b: i for i, b in enumerate(blocks)}
        TABLE = -1
        HELD = -2
        NONE = -1  # holding marker

        supports = [TABLE] * len(blocks)
        for b, sup in on_map.items():
            i = idx[b]
            if sup == "table":
                supports[i] = TABLE
            elif sup == "held":
                supports[i] = HELD
            else:
                supports[i] = idx[sup]

        hold = NONE if holding is None else idx[holding]

        goal_checks = []
        for t, a, b in goals:
            if t == "on":
                goal_checks.append((idx[a], idx[b]))
            else:
                goal_checks.append((idx[a], TABLE))

        def is_goal(supports_tup, hold_i):
            for a_i, sup_i in goal_checks:
                if supports_tup[a_i] != sup_i:
                    return False
            return True

        start = (tuple(supports), hold)
        if is_goal(start[0], start[1]):
            return []

        def top_blocks(supports_tup):
            has_on_top = [False] * len(blocks)
            for i, sup in enumerate(supports_tup):
                if sup >= 0:
                    has_on_top[sup] = True
            tops = [i for i in range(len(blocks)) if (not has_on_top[i]) and supports_tup[i] != HELD]
            return tops

     
        def gen_actions(supports_tup, hold_i):
            tops = top_blocks(supports_tup)

            if hold_i == NONE:
                engages = []
                unmounts = []
                for i in tops:
                    sup = supports_tup[i]
                    if sup == TABLE:
                        engages.append(("engage_payload", i, None))
                    elif sup >= 0:
                        unmounts.append(("unmount_node", i, sup))
                engages.sort(key=lambda a: blocks[a[1]])
                unmounts.sort(key=lambda a: (blocks[a[1]], blocks[a[2]]))
                return engages + unmounts

            releases = [("release_payload", hold_i, None)]
            mounts = []
            for j in tops:
                if j != hold_i:
                    mounts.append(("mount_node", hold_i, j))
            mounts.sort(key=lambda a: blocks[a[2]])
            return releases + mounts

        def apply_action(supports_tup, hold_i, act):
            name, a, b = act
            sup = list(supports_tup)

            if name == "engage_payload":
                sup[a] = HELD
                return (tuple(sup), a)

            if name == "unmount_node":
                sup[a] = HELD
                return (tuple(sup), a)

            if name == "release_payload":
                sup[a] = TABLE
                return (tuple(sup), NONE)

            if name == "mount_node":
                sup[a] = b
                return (tuple(sup), NONE)

            raise ValueError("Unknown action")

        q = deque([start])
        parent = {start: None}
        parent_act = {}

        while q:
            state = q.popleft()
            supports_tup, hold_i = state

            for act in gen_actions(supports_tup, hold_i):
                nxt = apply_action(supports_tup, hold_i, act)
                if nxt in parent:
                    continue
                parent[nxt] = state
                parent_act[nxt] = act

                if is_goal(nxt[0], nxt[1]):
                    out = []
                    cur = nxt
                    while parent[cur] is not None:
                        name, a, b = parent_act[cur]
                        if b is None:
                            out.append(f"({name} {blocks[a]})")
                        else:
                            out.append(f"({name} {blocks[a]} {blocks[b]})")
                        cur = parent[cur]
                    out.reverse()
                    return out

                q.append(nxt)

        return []

    # -------------------------
    # Domain 2: Objects world
    # -------------------------
    def _parse_objects_init(self, init_text: str):
        facts = self._split_facts(init_text)
        harmony = False
        planet = set()
        province = set()
        pain = set()
        craves = set()
        objs = set()

        for f in facts:
            f = f.strip().lower()
            if f == "harmony":
                harmony = True
                continue

            m = re.match(r"planet object (\w+)", f)
            if m:
                o = m.group(1)
                planet.add(o)
                objs.add(o)
                continue

            m = re.match(r"province object (\w+)", f)
            if m:
                o = m.group(1)
                province.add(o)
                objs.add(o)
                continue

            m = re.match(r"pain object (\w+)", f)
            if m:
                o = m.group(1)
                pain.add(o)
                objs.add(o)
                continue

            m = re.match(r"object (\w+) craves object (\w+)", f)
            if m:
                a, b = m.group(1), m.group(2)
                craves.add((a, b))
                objs.update([a, b])
                continue

        return harmony, planet, province, pain, craves, objs

    def _parse_objects_goal(self, goal_text: str):
        facts = self._split_facts(goal_text)
        goals = []
        for f in facts:
            f = f.strip().lower()
            m = re.match(r"object (\w+) craves object (\w+)", f)
            if m:
                goals.append((m.group(1), m.group(2)))
        return goals

    def _solve_objects(self, init_text: str, goal_text: str):
        harmony, planet_s, province_s, pain_s, craves_s, objs_s = self._parse_objects_init(init_text)
        goals = self._parse_objects_goal(goal_text)
        goal_set = set(goals)

        objs = sorted(set(objs_s) | set([x for x, y in goal_set] + [y for x, y in goal_set]))
        n = len(objs)
        idx = {o: i for i, o in enumerate(objs)}

        def bitset(items):
            m = 0
            for it in items:
                if it in idx:
                    m |= 1 << idx[it]
            return m

        planet = bitset(planet_s)
        province = bitset(province_s)
        pain = bitset(pain_s)

        craves = 0
        for a, b in craves_s:
            if a in idx and b in idx:
                craves |= 1 << (idx[a] * n + idx[b])

        goal_mask = 0
        for a, b in goal_set:
            goal_mask |= 1 << (idx[a] * n + idx[b])

        start = (harmony, planet, province, pain, craves)

        def is_goal(state):
            return (state[4] & goal_mask) == goal_mask

        if is_goal(start):
            return []

        def gen_actions(state):
            harm, pl, pr, pn, cr = state
            acts = []

            if harm:
                # attack
                for i in range(n):
                    bit = 1 << i
                    if (pr & bit) and (pl & bit):
                        acts.append(("attack", i, None))
                # feast
                for i in range(n):
                    bitx = 1 << i
                    if not (pr & bitx):
                        continue
                    base = i * n
                    for j in range(n):
                        if cr & (1 << (base + j)):
                            acts.append(("feast", i, j))

            for i in range(n):
                bitx = 1 << i
                if pn & bitx:
                    acts.append(("succumb", i, None))
                    for j in range(n):
                        bity = 1 << j
                        if pr & bity:
                            acts.append(("overcome", i, j))

            return acts

        def apply(state, act):
            harm, pl, pr, pn, cr = state
            name, i, j = act
            bitx = 1 << i

            if name == "attack":
                pn |= bitx
                pr &= ~bitx
                pl &= ~bitx
                harm = False
                return (harm, pl, pr, pn, cr)

            if name == "succumb":
                pr |= bitx
                pl |= bitx
                pn &= ~bitx
                harm = True
                return (harm, pl, pr, pn, cr)

            if name == "overcome":
                bity = 1 << j
                harm = True
                pr |= bitx
                cr |= 1 << (i * n + j)
                pr &= ~bity
                pn &= ~bitx
                return (harm, pl, pr, pn, cr)

            if name == "feast":
                pn |= bitx
                bity = 1 << j
                pr |= bity
                cr &= ~(1 << (i * n + j))
                pr &= ~bitx
                harm = False
                return (harm, pl, pr, pn, cr)

            raise ValueError("Unknown action")

        q = deque([start])
        parent = {start: None}
        parent_act = {}

        while q:
            s = q.popleft()
            for act in gen_actions(s):
                ns = apply(s, act)
                if ns in parent:
                    continue
                parent[ns] = s
                parent_act[ns] = act

                if is_goal(ns):
                    out = []
                    cur = ns
                    while parent[cur] is not None:
                        name, i, j = parent_act[cur]
                        if j is None:
                            out.append(f"({name} {objs[i]})")
                        else:
                            out.append(f"({name} {objs[i]} {objs[j]})")
                        cur = parent[cur]
                    out.reverse()
                    return out

                q.append(ns)


        return []
