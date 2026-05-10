_base_ = './default.py'
ModelParams = dict(
    override_w = 768,
)
OptimizationParams = dict(
    batch_size=2,
    iterations=18000,
    iters_per_step=1500,
    initial_skip_steps=8,
    camera_selection_batch_size=3,
    prompt="A realistic medium shot of a woman with blonde hair wearing a light yellow shirt in an apron slicing meat in a cluttered kitchen at night.", 
    source_prompt="A realistic medium shot of a man in an apron slicing meat in a cluttered kitchen at night.",
    guidance_scale=5.5,
)