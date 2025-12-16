
out_dir = 'out-instruct'
eval_interval = 200
eval_iters = 40
log_interval = 10
early_stop_patience = 3

init_from = 'resume' 


always_save_checkpoint = True

wandb_log = False
wandb_project = 'alpaca-sft'
wandb_run_name = 'gpt2-sft'

dataset = 'instruct_alpaca'
gradient_accumulation_steps = 2
batch_size = 10
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

learning_rate = 1e-4
max_iters = 22000
lr_decay_iters = 22000
min_lr = 1e-5
warmup_iters = 20
decay_lr = False

device = 'cuda'
compile = False
dtype = 'bfloat16'
