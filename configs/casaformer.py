_base_ = [
    './_base_/datasets/GBSS.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_80k.py',
    './casaformer_base.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=0.0, by_epoch=False)

runner = dict(max_iters=2000000)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=16, workers_per_gpu=2)
find_unused_parameters = True
