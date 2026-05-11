_base_ = './default.py'
ModelParams = dict(
    override_w = 768,
)
OptimizationParams = dict(
    batch_size=4,
    iterations=18000,
    initial_skip_steps=10,
    prompt="An iron man making a cocktail indoors, pouring a light blue drink from a metal shaker into a martini glass on a wooden table, wearing a blue denim cap and glasses, bright modern room, large windows with soft daylight, white shelf with bottles and books, cozy home bartender vlog, realistic photography, cinematic.", 
    source_prompt="A young man making a cocktail indoors, pouring a dark red drink from a metal shaker into a martini glass on a white table, wearing a blue denim cap and glasses, bright modern room, large windows with soft daylight, white shelf with bottles and books, cozy home bartender vlog, realistic photography, cinematic.",
    guidance_scale=12.5,
    step_sizes=[16, 8, 4, 4, 8]
)