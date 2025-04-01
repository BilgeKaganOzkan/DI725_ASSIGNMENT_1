# Configuration for training sentiment analysis model
# Default config values for sentiment analysis based on GPT-2 architecture

# Output directory
out_dir = 'pretrained-sentiment-gpt2'

# Evaluation settings
eval_interval = 250  # keep frequent because we'll overfit
validation_interval = 250  # validation interval, matches eval_interval by default
validation_at_epoch_end = True  # run validation at the end of each epoch
eval_iters = 200
log_interval = 50  # don't print too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

# Logging configuration
wandb_log = True  # override via command line if you like
wandb_project = '2697134-assignment1'
wandb_run_name = 'pretrained-sentiment-gpt2'

# Dataset configuration
dataset = 'subdata'

# Training settings
gradient_accumulation_steps = 2  # increased for larger model
batch_size = 16  # reduced for larger model
block_size = 256  # context of up to 256 previous characters

# GPT-2 architecture (using original model parameters)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1  # dropout for gpt2
sentiment_dropout = 0.2  # dropout for sentiment head

learning_rate = 5e-5  # lower learning rate for fine-tuning pretrained model
max_iters = 2000
lr_decay_iters = 3000  # longer decay schedule (beyond max_iters for sustained learning)
min_lr = 5e-6  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 400  # increased warmup period (20% of max_iters)

# Sentiment analysis specific
num_classes = 3  # negative, neutral, positive
sentiment_labels = ['negative', 'neutral', 'positive']
init_from = 'gpt2'  # initialize from pretrained GPT-2 model

compile = False