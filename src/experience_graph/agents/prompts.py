TEXTCRAFT_ACTION_GUIDE = """
TextCraft action rules:
- Choose exactly one action from available_actions and keep the same name and argument keys.
- Use the current task id and visible inventory/environment state to pick the shortest valid route.
- For unknown environment facts, use the matching inspect(...) or explore(...) action before relying on that route.
- If one route is unavailable, try another visible route while step budget remains.
- Use report_impossible(reason=no_viable_plan) only after visible crafting, mining, trading, inspection, exploration, loot, brewing, farming, combat, and waiting options relevant to the task are exhausted or blocked.
- Do not use hidden_facts. Only use facts shown in the observation or revealed by a prior action.
Return JSON only. next_action must be an object with name and args. Example JSON output: {"next_action":{"name":"move_to","args":{"location":"village"}},"reason":"short rationale","confidence":0.7}.
""".strip()
