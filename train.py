"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Modified to support sentiment analysis using CSV data from /data/subdata.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from transformers import GPT2Tokenizer  # Import GPT-2 tokenizer
import torch.nn.functional as F

from model.model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Custom dataset for sentiment analysis with CSV files
class SentimentDataset(Dataset):
    def __init__(self, csv_path, block_size=1024):
        self.df = pd.read_csv(csv_path)
        self.block_size = block_size
        
        print(f"Loading data from {csv_path}")
        
        # Initialize the GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # If needed, set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process the data
        self.examples = []
        self.labels = []
        
        for idx, row in self.df.iterrows():
            # Get prompt text
            if 'prompt' in row:
                text = str(row['prompt'])
                # Use GPT2 tokenizer instead of character-level tokenization
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,  # Add [CLS], [SEP]
                    max_length=block_size,    # Truncate to block_size
                    truncation=True           # Truncate to max_length
                )
                
                # Pad/truncate to exactly block_size tokens for consistent length
                if len(tokens) > block_size:
                    tokens = tokens[:block_size]
                else:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (block_size - len(tokens))
                
                self.examples.append(tokens)
                
                # Get sentiment label
                if 'customer_sentiment' in row:
                    sentiment = row['customer_sentiment']
                    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                    sentiment_idx = sentiment_map.get(sentiment, 1)  # Default to neutral
                    self.labels.append(sentiment_idx)
                else:
                    # Default to neutral if no label is provided
                    self.labels.append(1)
        
        print(f"Loaded {len(self.examples)} examples from {csv_path}")
        
        # Print some stats about token lengths
        lengths = [len(x) for x in self.examples]
        print(f"Average token length: {sum(lengths)/len(lengths):.1f}, Max: {max(lengths)}, Min: {min(lengths)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.examples[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# default config values designed to train a sentiment classifier
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # Wandb log enabled
wandb_project = 'sentiment-analysis' # Project name updated
wandb_run_name = 'sentiment_' + str(int(time.time())) # Unique run name
# validation specifics
validation_interval = 250  # Validation interval, now done every 250 iterations
validation_at_epoch_end = False  # Run validation at the end of each epoch
# data
dataset = 'subdata'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# Store the output directory path to ensure it's consistent throughout execution
checkpoint_dir = out_dir
print(f"Using configuration: init_from={init_from}, out_dir={out_dir}, checkpoint_dir={checkpoint_dir}")
print(f"Will save checkpoints to {os.path.abspath(checkpoint_dir)}")
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Setup data loaders for sentiment analysis
def get_data_loaders():
    data_dir = os.path.join('data', dataset)
    
    # Train dataset
    train_dataset = SentimentDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        block_size=block_size
    )
    
    # Validation dataset
    val_dataset = SentimentDataset(
        csv_path=os.path.join(data_dir, 'validation.csv'),
        block_size=block_size
    )
    
    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not ddp,
        sampler=torch.utils.data.distributed.DistributedSampler(train_dataset) if ddp else None,
        pin_memory=True,
    )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_loader, val_loader

# Load data
train_loader, val_loader = get_data_loaders()

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # Use GPT2 tokenizer vocabulary size instead of 256
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_args['vocab_size'] = len(tokenizer)  # GPT2 tokenizer vocabulary size
    model_args['num_classes'] = 3   # negative, neutral, positive
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {checkpoint_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'num_classes']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # Add num_classes for sentiment analysis
    model_args['num_classes'] = 3  # negative, neutral, positive
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Evaluate function for sentiment analysis
@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        with ctx:
            logits, loss = model(input_ids, labels)
        
        losses.append(loss.item())
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate class-specific metrics
    class_names = ['negative', 'neutral', 'positive']
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[f'precision_{class_name}'] = precision_per_class[i]
        class_metrics[f'recall_{class_name}'] = recall_per_class[i]
        class_metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    results = {
        'loss': torch.tensor(losses).mean().item(),
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'class_metrics': class_metrics,
        'all_preds': all_preds,
        'all_labels': all_labels
    }
    
    model.train()
    return results

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_loss = 0

print("Starting training")
model.train()

# Remove initial evaluation
# First evaluation at iter 0
if master_process:
    # results = evaluate()
    # print(f"step {iter_num}: val loss {results['loss']:.4f}, val accuracy {results['accuracy']:.4f}")
    
    # Save the initial checkpoint regardless of iter_num
    try:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        # Ensure output directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pt')
        print(f"Saving initial checkpoint to {checkpoint_path}")
        
        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Verify the checkpoint was saved successfully
        if os.path.exists(checkpoint_path):
            print(f"Initial checkpoint saved successfully: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        else:
            print(f"WARNING: Failed to save initial checkpoint to {checkpoint_path}")
    except Exception as e:
        print(f"ERROR saving initial checkpoint: {e}")
        import traceback
        traceback.print_exc()

if eval_only:
    print("eval_only mode, exiting")
    exit()

# Main training loop
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Inner training loop
    for batch in train_loader:
        # Get batch
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass through the model
        with ctx:
            logits, loss = model(input_ids, labels)
            loss = loss / gradient_accumulation_steps  # Scale for gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (iter_num + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Update loss tracking
        running_loss += loss.item() * gradient_accumulation_steps
        
        # Logging
        if iter_num % log_interval == 0 and iter_num > 0 and master_process:
            print(f"iter {iter_num}: loss {running_loss/log_interval:.4f}, lr {lr:.6f}")
            if wandb_log:
                wandb.log({
                    "train/loss": running_loss/log_interval,
                    "lr": lr,
                }, step=iter_num)
            running_loss = 0.0
        
        # Update iteration counters
        iter_num += 1
        local_iter_num += 1

        # Perform validation if iter_num is a multiple of validation_interval and iter_num > 0
        if iter_num % validation_interval == 0 and iter_num > 0 and master_process:
            print(f"\n--- Validation at iteration {iter_num} ---")
            results = evaluate()
            print(f"step {iter_num}: val loss {results['loss']:.4f}, val accuracy {results['accuracy']:.4f}")
            
            # Track best metrics
            best_metrics = {}
            if not hasattr(evaluate, 'best_metrics'):
                evaluate.best_metrics = {
                    'best_loss': float('inf'),
                    'best_accuracy': 0.0,
                    'best_f1': 0.0,
                    'best_iter': 0
                }
            
            # Update best results
            if results['loss'] < evaluate.best_metrics['best_loss']:
                evaluate.best_metrics['best_loss'] = results['loss'] 
                evaluate.best_metrics['best_iter'] = iter_num
            
            if results['accuracy'] > evaluate.best_metrics['best_accuracy']:
                evaluate.best_metrics['best_accuracy'] = results['accuracy']
                
            if results['f1_score'] > evaluate.best_metrics['best_f1']:
                evaluate.best_metrics['best_f1'] = results['f1_score']
            
            # Log metrics to wandb
            if wandb_log:
                # Log detailed validation metrics to wandb
                conf_matrix = results['confusion_matrix']
                class_names = ['negative', 'neutral', 'positive']
                
                # Add class-specific metrics to the log
                class_metrics = results['class_metrics']
                wandb_logs = {
                    "iter": iter_num,
                    "val/loss": results['loss'],
                    "val/accuracy": results['accuracy'],
                    "val/f1_score": results['f1_score'],
                    "lr": lr,
                    "progress": iter_num / max_iters * 100,  # Progress percentage
                    
                    # Include best values
                    "best/loss": evaluate.best_metrics['best_loss'],
                    "best/accuracy": evaluate.best_metrics['best_accuracy'],
                    "best/f1": evaluate.best_metrics['best_f1'],
                    "best/iter": evaluate.best_metrics['best_iter']
                }
                
                # Add class metrics
                for metric_name, value in class_metrics.items():
                    wandb_logs[f"val/{metric_name}"] = value
                
                # Log confusion matrix values
                for i, class_name_i in enumerate(class_names):
                    for j, class_name_j in enumerate(class_names):
                        wandb_logs[f"val/conf_matrix_{class_name_i}_{class_name_j}"] = conf_matrix[i][j]
                
                # Log confusion matrix visualization if available
                try:
                    if hasattr(wandb, 'plot'):
                        conf_plot = wandb.plot.confusion_matrix(
                            y_true=results['all_labels'],
                            preds=results['all_preds'],
                            class_names=class_names
                        )
                        wandb_logs["val/conf_matrix_plot"] = conf_plot
                except Exception as e:
                    print(f"Wandb confusion matrix visualization error: {e}")
                
                # Log all metrics to wandb
                print("Logging validation metrics to wandb...")
                # --- Debug Print Added ---
                print(f"[DEBUG] Logging to wandb at iter {iter_num}: {wandb_logs}")
                # --- Debug Print End ---
                wandb.log(wandb_logs, step=iter_num)
                print("Wandb logging completed.")
            
            # Save checkpoint
            try:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': evaluate.best_metrics['best_loss'],
                    'config': config,
                }
                
                # Ensure output directory exists
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                
                # Save the checkpoint
                torch.save(checkpoint, checkpoint_path)
                
                # Verify the checkpoint was saved successfully
                if os.path.exists(checkpoint_path):
                    print(f"Checkpoint saved successfully: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
                else:
                    print(f"WARNING: Failed to save checkpoint to {checkpoint_path}")
            except Exception as e:
                print(f"ERROR saving checkpoint: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"--- End of validation at iteration {iter_num} ---\n")
        
        # Check termination condition for the inner loop
        if iter_num >= max_iters:
            break
    
    # Check termination condition for the outer loop
    if iter_num >= max_iters:
        break

# Final evaluation
if master_process:
    results = evaluate()
    print(f"Final results: val loss {results['loss']:.4f}, val accuracy {results['accuracy']:.4f}, val f1 {results['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nClass-specific metrics:")
    for class_name in ['negative', 'neutral', 'positive']:
        print(f"  {class_name.capitalize()} - ",
              f"Precision: {results['class_metrics'][f'precision_{class_name}']:.4f}, ",
              f"Recall: {results['class_metrics'][f'recall_{class_name}']:.4f}, ",
              f"F1: {results['class_metrics'][f'f1_{class_name}']:.4f}")

if ddp:
    destroy_process_group()