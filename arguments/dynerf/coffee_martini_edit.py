_base_ = './default.py'
ModelParams = dict(
    override_w = 768,
)
OptimizationParams = dict(
    batch_size=4,
    iterations=18000,
    initial_skip_steps=10,
    camera_selection_batch_size=3,
    prompt="A man cooking a cabbage on a wooden table in a warm kitchen.", 
    source_prompt="A man cooking a steak in a warm kitchen.",
    guidance_scale=7.5,
)