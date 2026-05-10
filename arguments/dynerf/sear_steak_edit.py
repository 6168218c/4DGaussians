_base_ = './default.py'
ModelParams = dict(
    override_w = 768,
)
OptimizationParams = dict(
    batch_size=4,
    iterations=18000,
    iters_per_step=1500,
    initial_skip_steps=8,
    camera_selection_batch_size=3,
    prompt="An man in blue sleeves cooking a soup.", 
    source_prompt="A man in an apron cooking a steak.",
    guidance_scale=5.5,
)