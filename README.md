# Sentiment Analysis in Customer Service Conversations: Comparing Fine-Tuning and From-Scratch Training Approaches with nanoGPT

This project compares two transformer-based approaches for automatically classifying sentiment (positive, negative, neutral) in customer service conversations: a nanoGPT model trained from scratch and a fine-tuned GPT-2 model.

## Project Summary

Sentiment analysis in customer service conversations is a critical tool for companies to evaluate customer satisfaction and improve service quality. In this project, two different approaches were implemented to examine the performance of transformer architectures for this task:
1. A nanoGPT model trained from scratch
2. Fine-tuning a pre-trained GPT-2 model

The results demonstrate that transfer learning (fine-tuning) is a more effective strategy for sentiment analysis, especially with imbalanced datasets.

## Requirements

To run this project, you will need the following libraries:

```
torch>=1.10.0
transformers>=4.18.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
wandb>=0.12.0
```

To install the required libraries:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses a dataset containing customer service conversations. The dataset includes various features such as customer sentiment (positive, negative, neutral), topic area, and product category.

Dataset characteristics:
- Neutral: 55.2%
- Negative: 42.1%
- Positive: 2.7%

### Data Preprocessing

To prepare the dataset:

```bash
python data_preprocessing/preprocess_for_llm.py
```

This script:
1. Extracts customer utterances from conversation texts
2. Cleans and tokenizes the data
3. Splits the data into training and validation subsets

## Project Structure

```
├── config/                       # Training configuration files
│   ├── train_scratch.py          # From-scratch training configuration
│   └── train_pre_weight.py       # Fine-tuning configuration
├── data/                         # Data files
│   ├── original/                 # Original dataset
│   └── subdata/                  # Processed dataset
├── data_preprocessing/           # Data preprocessing scripts
│   ├── preprocess_for_llm.py     # Main preprocessing script
│   ├── data_explorer.py          # Data analysis script
│   └── data_exploration/         # Data analysis results
├── model/                        # Model definition and related code
│   └── model.py                  # NanoGPT model architecture
├── test_results/                 # Test results
│   ├── *_confusion_matrix.png    # Confusion matrix visualizations
│   └── *_results.txt             # Test metrics and results
├── wandb/                        # Weights & Biases log files
├── train.py                      # Main training script
├── test.py                       # Model evaluation script
└── README.md                     # This file
```

## Models

### Model Trained from Scratch
- 12 transformer layers
- 12 attention heads
- 768 hidden dimension
- 15K training iterations

### Fine-tuned Model (GPT-2)
- 12 transformer layers
- 12 attention heads
- 768 hidden dimension
- 15K training iterations

## Usage

### Training

To train a model from scratch:

```bash
python train.py --config config/train_scratch.py --out_dir scratch-sentiment-gpt2
```

To fine-tune a GPT-2 model:

```bash
python train.py --config config/train_pre_weight.py --out_dir pretrained-sentiment-gpt2
```

### Testing

To test the trained models:

```bash
python test.py scratch-sentiment-gpt2/ckpt.pt pretrained-sentiment-gpt2/ckpt.pt --test_csv_path data/subdata/test.csv --results_dir test_results
```

This command tests both models and saves the results to the `test_results` directory.

### Weights & Biases Integration

Weights & Biases (wandb) is used to monitor the training process. To configure wandb:

```bash
wandb login
```

You can track results in your own project by adjusting the `wandb_project` and `wandb_run_name` parameters during training.

## Results

### Model Trained from Scratch
- Accuracy: 66.67%
- Weighted F1-Score: 59.86%
- Positive F1-Score: 18% (1 out of 10 examples correct)

### Fine-tuned Model
- Accuracy: 90.00%
- Weighted F1-Score: 89.77%
- Positive F1-Score: 82% (7 out of 10 examples correct)

The fine-tuned GPT-2 model performed significantly better, especially on the under-represented positive class, demonstrating that fine-tuning a pre-trained model is more effective on imbalanced datasets.

## Recommendations and Future Work

1. Create a more balanced dataset using data augmentation techniques
2. Experiment with different pre-trained models such as BERT and RoBERTa
3. Develop model architectures that consider the structural features of customer-agent interactions

## Contributors

- Bilge Kağan ÖZKAN - Middle East Technical University