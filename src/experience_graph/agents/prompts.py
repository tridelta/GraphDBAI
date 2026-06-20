TEXTCRAFT_ACTION_GUIDE = """
TextCraft action rules:
- Choose only one action from available_actions and copy its name and args schema.
- craft(diamond armor) requires inventory.crafting_table=true and enough diamonds for that piece: helmet 5, chestplate 8, leggings 7, boots 4.
- craft(crafting_table) requires inventory.wood>=4.
- move_to(mine) requires environment.nearby_mine=true. If nearby_mine is unknown and mine_search_available=true, use explore(mine) before moving or mining.
- mine(diamond) requires environment.nearby_mine=true and an iron-or-better pickaxe.
- move_to(village) requires environment.nearby_village=true.
- inspect(village) requires being able to reach a known village; use it when village_has_armorer is unknown.
- trade with armorer requires location=village, village_has_armorer=true, and enough emeralds: diamond_set 40, one armor piece 10.
- If one route fails or is unavailable, try a visible alternative route while step budget remains, such as switching between trading, mining, crafting, inspection, and exploration.
- Use report_impossible(reason=no_viable_plan) only after visible crafting, mining, trading, inspection, and exploration options are exhausted or blocked. Do not report impossible just because one route failed.
- If the state already supports direct crafting or trading, prefer the shorter valid route.
Return JSON only. next_action must be an object with name and args. Example JSON output: {"next_action":{"name":"move_to","args":{"location":"village"}},"reason":"short rationale","confidence":0.7}.
""".strip()

