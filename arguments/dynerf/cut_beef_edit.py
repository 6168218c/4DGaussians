_base_ = './default.py'
ModelParams = dict(
    override_w = 768,
)
OptimizationParams = dict(
    batch_size=2,
    prompt="A realistic medium shot of a woman wearing a light yellow shirt in an apron slicing meat in a cluttered kitchen at night.", 
    source_prompt="A realistic medium shot of a man in an apron slicing meat in a cluttered kitchen at night.",
)