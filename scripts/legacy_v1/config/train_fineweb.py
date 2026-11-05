
out_dir = 'out-fineweb'
eval_interval = 500
log_interval = 10
eval_iters = 200
wandb_log = True
wandb_project = 'fineweb-gpt2'
wandb_run_name = 'gpt2-124M-10B'

dataset = 'fineweb'
init_from = 'resume'

batch_size = 10
block_size = 1024
gradient_accumulation_steps = 48

max_iters = 20000
lr_decay_iters = 20000

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000

device = 'cuda'
compile = False
dtype = 'bfloat16'
