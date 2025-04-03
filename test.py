import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
from transformers import GPT2Tokenizer
import argparse
import numpy as np
from contextlib import nullcontext
import time
import wandb # Import wandb
import re # Import re for sanitizing filenames
# Import matplotlib for confusion matrix visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming model.py is in a 'model' subdirectory relative to test.py
# Adjust the import path if your directory structure is different
from model.model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
TEST_CSV_PATH = 'data/subdata/test.csv'
CLASS_NAMES = ['negative', 'neutral', 'positive'] # Must match training label order
RESULTS_DIR = 'test_results' # Directory to save results files
WANDB_PROJECT = '2697134-assignment1' # Set your wandb project name

# -----------------------------------------------------------------------------
# Custom dataset for sentiment analysis (adapted from train.py)
class SentimentTestDataset(Dataset):
    def __init__(self, csv_path, tokenizer, block_size):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.block_size = block_size

        print(f"Loading test data from {csv_path}")

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.examples = []
        self.labels = []

        for idx, row in self.df.iterrows():
            if 'prompt' in row and 'customer_sentiment' in row:
                text = str(row['prompt'])
                sentiment = str(row['customer_sentiment']).lower() # Ensure lowercase

                # Encode text
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self.block_size,
                    truncation=True
                )

                # Pad/truncate tokens
                if len(tokens) > self.block_size:
                    tokens = tokens[:self.block_size]
                else:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (self.block_size - len(tokens))

                self.examples.append(tokens)

                # Map sentiment label
                sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                sentiment_idx = sentiment_map.get(sentiment, -1) # Use -1 for unknown labels

                if sentiment_idx != -1:
                    self.labels.append(sentiment_idx)
                else:
                    print(f"Warning: Unknown sentiment label '{sentiment}' at row {idx}. Skipping.")
                    # Remove the corresponding example if label is invalid
                    self.examples.pop()

        if not self.examples:
             raise ValueError(f"No valid examples found in {csv_path}. Check 'prompt' and 'customer_sentiment' columns and label values.")

        print(f"Loaded {len(self.examples)} valid test examples from {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.examples[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Evaluation function (adapted from train.py)
@torch.no_grad()
def evaluate_model(model, loader, device, ctx):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Starting evaluation on {device}...")
    batch_num = 0
    total_batches = len(loader)
    for batch in loader:
        batch_num += 1
        print(f"  Processing batch {batch_num}/{total_batches}...", end='\r') # Use \r for cleaner output
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device) # Keep labels for metric calculation

        with ctx:
            # Ensure model returns logits even if labels are not provided during eval
            logits, _ = model(input_ids) # We don't need loss for testing

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print("\nEvaluation complete. Calculating metrics...") # Newline after processing batches

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') # Use weighted for potential class imbalance
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    # Detailed classification report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)

    results = {
        'accuracy': accuracy,
        'f1_score_weighted': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

    model.train() # Set back to train mode just in case
    return results

# -----------------------------------------------------------------------------
# Helper function to format confusion matrix for printing/saving
def format_confusion_matrix(cm, class_names):
    header = "       " + "  ".join([f"{name:<10}" for name in class_names])
    lines = [header]
    for i, row in enumerate(cm):
        line = f"{class_names[i]:<10}" + " ".join([f"{count:<10}" for count in row])
        lines.append(line)
    separator = "-" * (12 + 11 * len(class_names))
    return "\n".join(lines + [separator])

# Helper function to sanitize filename
def sanitize_filename(name):
    # Remove potentially problematic characters for filenames
    name = re.sub(r'[<>:"/\\|?*]+', '_', name)
    # Remove leading/trailing whitespace and replace spaces with underscores
    name = name.strip().replace(' ', '_')
    # Ensure it's not empty
    if not name:
        name = "unnamed_model"
    return name

# Add a function to create and save confusion matrix visualization
def plot_confusion_matrix(cm, class_names, save_path):
    """
    Generate a confusion matrix visualization and save it to a file.
    
    Args:
        cm: The confusion matrix array
        class_names: List of class names
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Use seaborn's heatmap for better visualization
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Add labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure with tight layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix visualization saved to {save_path}")

# -----------------------------------------------------------------------------
# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Sentiment GPT models.')
    parser.add_argument('checkpoint_paths', nargs='+', help='Path(s) to checkpoint file(s) (.pt)')
    parser.add_argument('--test_csv_path', type=str, default=TEST_CSV_PATH, help='Path to the test CSV file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing (adjust based on GPU memory)')
    parser.add_argument('--wandb_project', type=str, default=WANDB_PROJECT, help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=f'test_run_{int(time.time())}', help='Weights & Biases run name')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='Directory to save result files')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')


    args = parser.parse_args()

    # --- Setup Results Directory ---
    try:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(args.results_dir)}")
    except OSError as e:
        print(f"ERROR: Could not create results directory '{args.results_dir}': {e}")
        # Optionally exit if directory cannot be created
        # exit(1)


    # --- Initialize Wandb (once per script run) ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, job_type="evaluation")
            print(f"Wandb initialized for run: {args.wandb_run_name} in project: {args.wandb_project}")
        except Exception as e:
            print(f"ERROR: Failed to initialize wandb: {e}. Disabling wandb.")
            use_wandb = False
    else:
        print("Wandb logging disabled.")


    print(f"Using device: {args.device}")
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    # Ensure dtype is explicitly handled based on device capability for inference
    # Using float32 is generally safer for inference unless specific precision is needed and supported
    ptdtype = torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) # Match dtype for autocast if needed

    # --- Initialize Tokenizer ---
    print("Initializing tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized.")
    except Exception as e:
        print(f"ERROR initializing tokenizer: {e}")
        exit(1) # Exit if tokenizer fails


    # --- Test Each Model ---
    all_model_results = {}

    for ckpt_path in args.checkpoint_paths:
        # --- Derive and Sanitize Model Name ---
        try:
            # Try getting the parent directory name first
            parent_dir = os.path.dirname(ckpt_path)
            model_name = os.path.basename(parent_dir) if parent_dir and parent_dir != '.' and parent_dir != '/' else None

            # If parent dir name is not suitable, use the filename without extension
            if not model_name:
                model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

            # Sanitize the name for use in filenames and logging
            model_name = sanitize_filename(model_name)
            print(f"Derived and sanitized model name: {model_name}")

        except Exception as e:
            print(f"ERROR deriving model name from path '{ckpt_path}': {e}")
            model_name = sanitize_filename(f"unknown_model_{os.path.basename(ckpt_path)}")
            print(f"Using fallback model name: {model_name}")


        print(f"\n{'='*40}")
        print(f"Loading and testing model '{model_name}' from: {ckpt_path}")
        print(f"{'='*40}")

        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint file not found at {ckpt_path}. Skipping this model.")
            continue

        # --- Load Checkpoint ---
        print("Loading checkpoint...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=args.device)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint {ckpt_path}. Error: {e}. Skipping this model.")
            continue
        print("Checkpoint loaded.")

        # --- Load Config and Model Args ---
        if 'model_args' not in checkpoint:
             print(f"ERROR: 'model_args' not found in checkpoint {ckpt_path}. Cannot reconstruct model. Skipping this model.")
             continue
        model_args = checkpoint['model_args']

        block_size = model_args.get('block_size', 1024) # Default if not in args
        print(f"Using block_size from checkpoint: {block_size}")

        # --- Setup Dataset and DataLoader ---
        try:
            print(f"Setting up dataset using test CSV: {args.test_csv_path}")
            test_dataset = SentimentTestDataset(args.test_csv_path, tokenizer, block_size)
            if len(test_dataset) == 0:
                 print(f"WARNING: No valid test examples loaded from {args.test_csv_path}. Check file content and format.")
                 continue # Skip if dataset is empty
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                pin_memory=(args.device == 'cuda') # pin_memory only works with CUDA
            )
            print(f"DataLoader setup complete with {len(test_loader)} batches.")
        except FileNotFoundError:
             print(f"ERROR: Test CSV file not found at '{args.test_csv_path}'. Skipping this model.")
             continue
        except ValueError as e: # Catch specific ValueError from dataset init
             print(f"ERROR: Problem loading test data: {e}. Skipping this model.")
             continue
        except Exception as e:
            print(f"ERROR setting up dataset/loader: {e}. Skipping this model.")
            import traceback
            traceback.print_exc()
            continue


        # --- Initialize Model ---
        print("Initializing model...")
        try:
            # Ensure vocab_size matches tokenizer, crucial for embedding layer
            expected_vocab_size = len(tokenizer)
            if model_args.get('vocab_size') != expected_vocab_size:
                 print(f"Warning: Model vocab_size ({model_args.get('vocab_size')}) doesn't match tokenizer ({expected_vocab_size}). Overriding.")
                 model_args['vocab_size'] = expected_vocab_size

            # Ensure num_classes is present, default to 3 if needed
            if 'num_classes' not in model_args:
                 print(f"Warning: 'num_classes' not found in model_args. Assuming 3 classes.")
                 model_args['num_classes'] = 3

            # Set dropout to 0 for evaluation
            model_args['dropout'] = 0.0
            if 'sentiment_dropout' in model_args: model_args['sentiment_dropout'] = 0.0

            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            print("Model structure initialized.")
        except Exception as e:
            print(f"ERROR creating model instance: {e}. Skipping this model.")
            continue

        # --- Load Model State ---
        print("Loading model state dictionary...")
        state_dict = checkpoint['model']
        # Clean potential DDP prefix
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        try:
            model.load_state_dict(state_dict, strict=False) # Use strict=False first for flexibility
            print("Model state loaded (strict=False).")
            # Optional: Check for missing/unexpected keys if strict=False was needed
            # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            # if missing_keys: print(f"Warning: Missing keys: {missing_keys}")
            # if unexpected_keys: print(f"Warning: Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"ERROR loading model state_dict: {e}. Skipping this model.")
            continue


        model.to(args.device)
        print(f"Model moved to device: {args.device}")

        # --- Evaluate Model ---
        print("Starting model evaluation...")
        try:
            results = evaluate_model(model, test_loader, args.device, ctx)
            all_model_results[model_name] = results # Store for later comparison/summary
            print("Model evaluation completed.")
        except Exception as e:
            print(f"ERROR during evaluation for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping results saving and logging for this model.")
            continue # Skip to next model if evaluation fails

        # --- Print Results ---
        print("\n--- Test Results ---")
        print(f"Model: {model_name} ({ckpt_path})")
        print(f"Test Set: {args.test_csv_path}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_score_weighted']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        print("\nConfusion Matrix:")
        try:
            formatted_cm = format_confusion_matrix(results['confusion_matrix'], CLASS_NAMES)
            print(formatted_cm)
        except Exception as e:
            print(f"Error formatting confusion matrix: {e}")
            print(results['confusion_matrix']) # Print raw matrix if formatting fails


        # --- Save Results to File ---
        # Ensure model_name is valid before creating filename
        if not model_name:
             print("ERROR: Cannot save results because model_name is empty. Skipping file save.")
        else:
            results_filename = os.path.join(args.results_dir, f"{model_name}_results.txt")
            print(f"Attempting to save results to: {results_filename}")
            try:
                with open(results_filename, 'w', encoding='utf-8') as f: # Specify encoding
                    f.write(f"--- Test Results ---\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Checkpoint Path: {ckpt_path}\n")
                    f.write(f"Test Set: {args.test_csv_path}\n\n")
                    f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"F1 Score (Weighted): {results['f1_score_weighted']:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(results['classification_report'])
                    f.write("\n\nConfusion Matrix:\n")
                    f.write(formatted_cm) # Use the formatted one
                print(f"Results saved successfully to {results_filename}")
                
                # Save confusion matrix as image
                cm_image_path = os.path.join(args.results_dir, f"{model_name}_confusion_matrix.png")
                plot_confusion_matrix(
                    results['confusion_matrix'],
                    CLASS_NAMES,
                    cm_image_path
                )
            except Exception as e:
                # Print a more detailed error message
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR saving results to file '{results_filename}': {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                import traceback
                traceback.print_exc() # Print stack trace for debugging


        # --- Log Results to Wandb ---
        if use_wandb:
            print("Logging results to wandb...")
            try:
                wandb_test_logs = {}
                # Log main metrics
                wandb_test_logs[f"{model_name}/test/accuracy"] = results['accuracy']
                wandb_test_logs[f"{model_name}/test/f1_score_weighted"] = results['f1_score_weighted']

                # Log confusion matrix values
                cm = results['confusion_matrix']
                for i, class_name_i in enumerate(CLASS_NAMES):
                    for j, class_name_j in enumerate(CLASS_NAMES):
                        # Ensure the key is wandb-compatible (no invalid chars like '/')
                        log_key = f"{model_name}/test/conf_matrix/{class_name_i}_vs_{class_name_j}"
                        wandb_test_logs[log_key] = cm[i][j]

                # Log confusion matrix plot
                print(f"Preparing confusion matrix plot for wandb for model {model_name}...")
                try:
                    if hasattr(wandb, 'plot'): # Check if plot submodule exists
                        print(f"  wandb.plot submodule found.")
                        print(f"  Data types - Labels: {type(results['all_labels'])}, Preds: {type(results['all_preds'])}")
                        # Ensure data are lists or numpy arrays
                        y_true_data = results['all_labels']
                        preds_data = results['all_preds']
                        if isinstance(y_true_data, list) and isinstance(preds_data, list):
                            print(f"  Data lengths - Labels: {len(y_true_data)}, Preds: {len(preds_data)}")
                        elif isinstance(y_true_data, np.ndarray) and isinstance(preds_data, np.ndarray):
                             print(f"  Data shapes - Labels: {y_true_data.shape}, Preds: {preds_data.shape}")

                        wandb_cm = wandb.plot.confusion_matrix(
                            y_true=y_true_data,
                            preds=preds_data,
                            class_names=CLASS_NAMES
                            )
                        # Ensure plot key is wandb-compatible
                        plot_key = f"{model_name}/test/confusion_matrix_plot"
                        wandb_test_logs[plot_key] = wandb_cm
                        print(f"  Confusion matrix plot generated and added to logs with key: {plot_key}")
                    else:
                         print("  WARNING: wandb.plot not available for confusion matrix visualization.")
                except Exception as plot_e:
                     print(f"  ERROR: Wandb confusion matrix visualization error: {plot_e}")
                     import traceback
                     traceback.print_exc()

                # Log all collected metrics for this model
                print(f"Attempting wandb.log() for model {model_name}...")
                wandb.log(wandb_test_logs)
                print(f"Wandb logging completed for model {model_name}.")
            except Exception as wandb_e:
                 print(f"ERROR logging to wandb for model {model_name}: {wandb_e}")


        # --- Cleanup ---
        print(f"Cleaning up resources for model {model_name}...")
        del model, checkpoint, state_dict, results # Explicitly delete large objects
        if args.device == 'cuda':
            torch.cuda.empty_cache() # Clear CUDA cache
            print("CUDA cache cleared.")

    # --- Finalize Wandb ---
    if use_wandb:
        try:
            wandb.finish()
            print("Wandb run finished.")
        except Exception as e:
            print(f"Error finishing wandb run: {e}")


    print("\nAll specified models tested.")
