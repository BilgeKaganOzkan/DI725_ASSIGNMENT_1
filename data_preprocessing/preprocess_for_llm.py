#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation for LLM Fine-Tuning
-----------------------------------
This script prepares data for LLM fine-tuning for sentiment analysis in customer service conversations.

The following steps are performed:
1. Data Loading
2. Text Preprocessing
3. Prompt Creation for LLM
4. Dataset Balancing and Strategic Sampling
5. Creation of Train/Validation/Test Datasets (with prompt and all features used to create it)
"""

import pandas as pd
import numpy as np
import re
import os
import json
from sklearn.model_selection import StratifiedKFold
from transformers import GPT2Tokenizer
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Output directories
OUTPUT_DIR = 'data/subdata'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load raw data"""
    train_df = pd.read_csv('data/original/train.csv')
    test_df = pd.read_csv('data/original/test.csv')
    
    return train_df, test_df

def preprocess_text(text):
    """Enhanced text preprocessing with improved filtering"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with [URL] token
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    
    # Replace emails with [EMAIL] token
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Replace pure numeric tokens with [NUMBER] token (e.g., "123")
    text = re.sub(r'\b\d+\b', '[NUMBER]', text)
    
    # Replace order numbers (alphanumeric patterns) with [NUMBER] token
    # Examples: "bb123456", "order#12345", "order-123", etc.
    text = re.sub(r'\b[a-z]*\d+[a-z0-9]*\b', '[NUMBER]', text)
    
    # Replace specific patterns like "order number" followed by alphanumeric
    text = re.sub(r'order number is [a-z0-9#-]+', 'order number is [NUMBER]', text)
    text = re.sub(r'order number [a-z0-9#-]+', 'order number [NUMBER]', text)
    text = re.sub(r'order #[a-z0-9-]+', 'order #[NUMBER]', text)
    text = re.sub(r'order[- ]?id [a-z0-9#-]+', 'order-id [NUMBER]', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove extra punctuation (but keep basic marks)
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    
    return text

def format_conversation(conversation):
    """Format conversation with special tokens to help the model distinguish speakers"""
    if not isinstance(conversation, str):
        return ""
        
    lines = conversation.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('Customer:'):
                formatted_lines.append('[CUSTOMER] ' + line[9:].strip())
            elif line.startswith('Agent:'):
                formatted_lines.append('[AGENT] ' + line[6:].strip())
            else:
                # Add continuing lines to the previous speaker or handle unmarked lines
                if formatted_lines and '[CUSTOMER]' in formatted_lines[-1]:
                    formatted_lines[-1] += ' ' + line
                elif formatted_lines and '[AGENT]' in formatted_lines[-1]:
                    formatted_lines[-1] += ' ' + line
                # If no previous speaker exists, assume it's an agent line
                elif not formatted_lines:
                    formatted_lines.append('[AGENT] ' + line)
                else:
                    # If line doesn't have a speaker prefix but appears after other lines
                    # Try to determine if it's agent or customer based on pattern
                    if "thank you for calling" in line.lower() or "my name is" in line.lower():
                        formatted_lines.append('[AGENT] ' + line)
                    else:
                        # Default to customer if unclear
                        formatted_lines.append('[CUSTOMER] ' + line)
    
    # Ensure at least one agent and customer line exists
    has_customer = any('[CUSTOMER]' in line for line in formatted_lines)
    has_agent = any('[AGENT]' in line for line in formatted_lines)
    
    if not has_customer:
        formatted_lines.append('[CUSTOMER] Thank you for your assistance')
    
    if not has_agent:
        formatted_lines.insert(0, '[AGENT] How may I help you today?')
    
    return ' '.join(formatted_lines)

def extract_customer_text(conversation):
    """Extract only customer parts from the conversation"""
    if not isinstance(conversation, str):
        return ""
        
    lines = conversation.split('\n')
    customer_lines = []
    is_customer = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Customer:'):
            is_customer = True
            customer_text = line[9:].strip()
            if customer_text:  # If line is not empty
                customer_lines.append(customer_text)
        elif line.startswith('Agent:'):
            is_customer = False
        elif is_customer and line:  # If still customer talking and line is not empty
            customer_lines.append(line)
    
    return ' '.join(customer_lines)

def create_prompt_template(format_type="full"):
    """Create prompt template for LLM fine-tuning, incorporating key features from analysis"""
    if format_type == "full":
        prompt_template = """Analyze the following customer service conversation and determine the customer's sentiment.

Conversation: {conversation}

The customer's sentiment is: {sentiment}

This is a training example for sentiment analysis in customer service conversations.
"""
    elif format_type == "conversation_only":
        prompt_template = """Analyze the following customer service conversation and determine the customer's sentiment.

Conversation: {conversation}

The customer's sentiment is: {sentiment}

This is a training example for sentiment analysis in customer service conversations.
"""
    elif format_type == "customer_only":
        prompt_template = """Analyze the following customer message and determine the customer's sentiment.

Customer Message: {customer_text}

The customer's sentiment is: {sentiment}

This is a training example for sentiment analysis in customer service conversations.
"""
    elif format_type == "few_shot":
        prompt_template = """Analyze the following customer service conversations and determine the customer's sentiment.

Example 1:
Conversation: [CUSTOMER] Hello, I still haven't received the product I ordered [AGENT] Hello, may I have your order number? [CUSTOMER] Sure, my order number is #12345 [AGENT] Thank you, your order has been shipped and should arrive tomorrow [CUSTOMER] Okay, thank you
Customer's sentiment: neutral

Example 2:
Conversation: [CUSTOMER] This product is not at all what I expected, it's very poor quality. I want my money back! [AGENT] I'm sorry for the problem you're experiencing. We can start the refund process [CUSTOMER] Yes please, I'm very dissatisfied with this situation
Customer's sentiment: negative

Example 3:
Conversation: [CUSTOMER] The product arrived on time and was better quality than I expected. Thank you very much! [AGENT] Thank you for your satisfaction, is there anything else we can help you with? [CUSTOMER] No, everything is perfect
Customer's sentiment: positive

Now analyze this conversation:
Conversation: {conversation}

The customer's sentiment is: {sentiment}
"""
    elif format_type == "feature_enhanced":
        # Enhanced prompt that includes key indicators
        prompt_template = """Analyze the following customer service conversation and determine the customer's sentiment.

Conversation: {conversation}

Conversation Length: {conv_length} characters
Customer Question Marks: {question_marks}
Customer Turns: {customer_turns}

The customer's sentiment is: {sentiment}

This is a training example for sentiment analysis in customer service conversations.
"""
    
    return prompt_template

def create_prompt(row, prompt_template):
    """Create a prompt for a given row using the template"""
    # Basic data preparation
    conversation = row['formatted_conversation'] if 'formatted_conversation' in row else row['conversation']
    customer_text = row['customer_text'] if 'customer_text' in row else ""
    sentiment = row['customer_sentiment']
    
    # Advanced features for enhanced prompts
    conv_length = len(conversation) if 'formatted_conversation' in row else len(row['conversation'])
    
    # Count question marks in customer text
    question_marks = customer_text.count('?') if customer_text else 0
    
    # Count customer turns
    customer_turns = len([l for l in (row['conversation'] if isinstance(row['conversation'], str) else "").split('\n') 
                         if l.strip().startswith('Customer:')])
    
    # Fill template
    prompt = prompt_template.format(
        conversation=conversation,
        customer_text=customer_text,
        sentiment=sentiment,
        conv_length=conv_length,
        question_marks=question_marks,
        customer_turns=customer_turns
    )
    
    return prompt

def validate_prompt(prompt, sentiment):
    """Validate that the prompt is correctly formatted for LLM training"""
    # Check if prompt contains the conversation
    if "Conversation:" not in prompt:
        return False, "Missing 'Conversation:' section"
    
    # Check if the prompt contains the expected sentiment
    expected_text = f"The customer's sentiment is: {sentiment}"
    if expected_text not in prompt:
        return False, f"Missing '{expected_text}' in prompt"
    
    # Check for any unprocessed order numbers
    order_number_patterns = [
        r"order number is \w+\d+\w*",
        r"order number \w+\d+\w*",
        r"order #\w+\d+\w*"
    ]
    
    for pattern in order_number_patterns:
        matches = re.findall(pattern, prompt.lower())
        if matches:
            return False, f"Unprocessed order number found: {matches[0]}"
    
    return True, "Prompt is valid for LLM training"

def calculate_sample_weights(train_df):
    """Calculate weights for strategic sampling based on class imbalance and key features"""
    # Class imbalance weights
    sentiment_counts = train_df['customer_sentiment'].value_counts()
    max_count = sentiment_counts.max()
    sentiment_weights = {sentiment: max_count / count for sentiment, count in sentiment_counts.items()}
    
    # Conversation features weights
    if 'conversation_length' in train_df.columns:
        mean_length = train_df['conversation_length'].mean()
        std_length = train_df['conversation_length'].std()
    else:
        train_df['conversation_length'] = train_df['conversation'].apply(len)
        mean_length = train_df['conversation_length'].mean()
        std_length = train_df['conversation_length'].std()
    
    # Calculate weight for each example
    def get_weight(row):
        # Base weight: Class weight
        weight = sentiment_weights[row['customer_sentiment']]
        
        # Additional weight if example is far from mean (could be important)
        length_zscore = (row['conversation_length'] - mean_length) / std_length
        if abs(length_zscore) > 1.5:
            weight *= 1.2
        
        # Additional weight if customer message is long (may contain more sentiment signals)
        if 'customer_text' in row and len(row['customer_text']) > 100:
            weight *= 1.1
            
        # Special case for the minority class (positive)
        if row['customer_sentiment'] == 'positive':
            weight *= 1.5  # Additional boost for the very rare positive class
            
        return weight
    
    train_df['sample_weight'] = train_df.apply(get_weight, axis=1)
    
    return train_df

def prepare_data_for_llm(train_df, test_df, prompt_format="few_shot"):
    """Prepare data for LLM fine-tuning with enhanced preprocessing"""
    # Format conversations
    train_df['formatted_conversation'] = train_df['conversation'].apply(format_conversation)
    test_df['formatted_conversation'] = test_df['conversation'].apply(format_conversation)
    
    # Preprocess each formatted conversation
    train_df['formatted_conversation'] = train_df['formatted_conversation'].apply(preprocess_text)
    test_df['formatted_conversation'] = test_df['formatted_conversation'].apply(preprocess_text)
    
    # Extract only customer texts
    train_df['customer_text'] = train_df['conversation'].apply(extract_customer_text)
    test_df['customer_text'] = test_df['conversation'].apply(extract_customer_text)
    
    # Preprocess customer texts
    train_df['customer_text'] = train_df['customer_text'].apply(preprocess_text)
    test_df['customer_text'] = test_df['customer_text'].apply(preprocess_text)
    
    # Basic feature engineering - these are needed for sample weights and prompt creation
    train_df['conversation_length'] = train_df['formatted_conversation'].apply(len)
    test_df['conversation_length'] = test_df['formatted_conversation'].apply(len)
    
    train_df['customer_text_length'] = train_df['customer_text'].apply(len)
    test_df['customer_text_length'] = test_df['customer_text'].apply(len)
    
    # Extract distinctive features - needed for prompt creation in feature_enhanced mode
    train_df['customer_question_marks'] = train_df['customer_text'].apply(lambda x: x.count('?'))
    test_df['customer_question_marks'] = test_df['customer_text'].apply(lambda x: x.count('?'))
    
    train_df['customer_exclamation_marks'] = train_df['customer_text'].apply(lambda x: x.count('!'))
    test_df['customer_exclamation_marks'] = test_df['customer_text'].apply(lambda x: x.count('!'))
    
    # Calculate customer and agent turns - needed for prompt creation in feature_enhanced mode
    train_df['customer_turns'] = train_df['conversation'].apply(
        lambda x: len([l for l in (x.split('\n') if isinstance(x, str) else []) if l.strip().startswith('Customer:')])
    )
    test_df['customer_turns'] = test_df['conversation'].apply(
        lambda x: len([l for l in (x.split('\n') if isinstance(x, str) else []) if l.strip().startswith('Customer:')])
    )
    
    train_df['agent_turns'] = train_df['conversation'].apply(
        lambda x: len([l for l in (x.split('\n') if isinstance(x, str) else []) if l.strip().startswith('Agent:')])
    )
    test_df['agent_turns'] = test_df['conversation'].apply(
        lambda x: len([l for l in (x.split('\n') if isinstance(x, str) else []) if l.strip().startswith('Agent:')])
    )
    
    # Create prompt template based on format
    if prompt_format == "feature_enhanced" and 'customer_question_marks' in train_df.columns:
        # Use the feature-enhanced prompt if we have the required features
        prompt_template = create_prompt_template("feature_enhanced")
    else:
        prompt_template = create_prompt_template(prompt_format)
    
    # Create prompts
    train_df['prompt'] = train_df.apply(lambda row: create_prompt(row, prompt_template), axis=1)
    test_df['prompt'] = test_df.apply(lambda row: create_prompt(row, prompt_template), axis=1)
    
    # Validate prompts for LLM training
    print("Validating prompts for LLM training...")
    invalid_prompts = 0
    for idx, row in train_df.sample(min(10, len(train_df))).iterrows():
        is_valid, message = validate_prompt(row['prompt'], row['customer_sentiment'])
        if not is_valid:
            print(f"Invalid prompt at index {idx}: {message}")
            print("First 200 chars:", row['prompt'][:200], "...")
            invalid_prompts += 1
    
    validation_message = f"Prompt validation complete. Found {invalid_prompts} invalid prompts in sample."
    print(validation_message)
    
    if invalid_prompts > 0:
        print("WARNING: Some prompts may not be properly formatted for LLM training.")
    else:
        print("All sampled prompts are valid for LLM training.")
    
    # Calculate weights for strategic sampling
    train_df = calculate_sample_weights(train_df)
    
    # Create train and validation sets (weighted stratified split)
    # First oversample (for balanced class distribution)
    sentiment_counts = train_df['customer_sentiment'].value_counts()
    max_class_count = sentiment_counts.max()
    
    # Use k-fold stratified sampling for more reliable splitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(skf.split(train_df, train_df['customer_sentiment']))
    
    train_set = train_df.iloc[train_indices]
    val_set = train_df.iloc[val_indices]
    
    # IMPORTANT: Make sure test data remains separate - never use test data for training
    # Check for potential data leakage
    train_conversations = set(train_set['conversation'].dropna())
    val_conversations = set(val_set['conversation'].dropna())
    test_conversations = set(test_df['conversation'].dropna())
    
    # Verify no overlap between train/val and test
    train_test_overlap = train_conversations.intersection(test_conversations)
    val_test_overlap = val_conversations.intersection(test_conversations)
    
    if train_test_overlap:
        # Remove any test examples from train if they somehow got there
        train_set = train_set[~train_set['conversation'].isin(test_conversations)]
    
    if val_test_overlap:
        # Remove any test examples from validation if they somehow got there
        val_set = val_set[~val_set['conversation'].isin(test_conversations)]
    
    # Balanced sampling for each class
    balanced_train_sets = []
    for sentiment in train_df['customer_sentiment'].unique():
        class_df = train_set[train_set['customer_sentiment'] == sentiment]
        
        # Sort examples by weight
        class_df = class_df.sort_values('sample_weight', ascending=False)
        
        # Take the same number of examples for each class
        target_count = min(max_class_count, len(class_df) * 3)  # Sample at least 3x
        if len(class_df) < target_count:
            # Fill in missing examples with weighted random sampling
            samples_needed = target_count - len(class_df)
            extra_samples = class_df.sample(n=samples_needed, replace=True, weights='sample_weight')
            class_df = pd.concat([class_df, extra_samples])
        else:
            # Take the highest weighted examples to target count
            class_df = class_df.head(target_count)
        
        balanced_train_sets.append(class_df)
    
    # Create balanced training set
    balanced_train_set = pd.concat(balanced_train_sets)
    balanced_train_set = balanced_train_set.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Keep all columns that are used to create the prompt plus the prompt itself
    columns_to_keep = [
        'customer_sentiment',          # The sentiment label (target)
        'prompt',                      # The LLM-ready formatted prompt (input)
        'conversation',                # Original conversation
        'formatted_conversation',      # Formatted conversation with speaker tokens
        'customer_text',               # Extracted customer text
        'conversation_length',         # Conversation length feature
        'customer_text_length',        # Customer text length feature
        'customer_question_marks',     # Question marks count feature
        'customer_exclamation_marks',  # Exclamation marks count feature
        'customer_turns',              # Number of customer turns feature
        'agent_turns',                 # Number of agent turns feature
        'sample_weight'                # Sample weight for potential resampling
    ]
    
    # Filter to keep all columns that exist
    balanced_train_set = balanced_train_set[
        [col for col in columns_to_keep if col in balanced_train_set.columns]
    ]
    val_set = val_set[
        [col for col in columns_to_keep if col in val_set.columns]
    ]
    test_df = test_df[
        [col for col in columns_to_keep if col in test_df.columns]
    ]
    
    # Save datasets
    balanced_train_set.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(OUTPUT_DIR, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Final validation - create an example file with samples of prompts
    with open(os.path.join(OUTPUT_DIR, 'prompt_examples.txt'), 'w', encoding='utf-8') as f:
        f.write("# Example Prompts for LLM Training\n\n")
        for sentiment in ['positive', 'negative', 'neutral']:
            examples = balanced_train_set[balanced_train_set['customer_sentiment'] == sentiment].sample(min(3, len(balanced_train_set[balanced_train_set['customer_sentiment'] == sentiment])))
            for _, row in examples.iterrows():
                f.write(f"## {sentiment.upper()} Example\n\n")
                f.write(f"```\n{row['prompt']}\n```\n\n")
    
    print(f"Saved {len(balanced_train_set)} training examples, {len(val_set)} validation examples, and {len(test_df)} test examples.")
    print(f"Example prompts saved to {os.path.join(OUTPUT_DIR, 'prompt_examples.txt')}")
    
    return balanced_train_set, val_set, test_df

def main():
    """Main workflow"""
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Step 2: Data preparation for LLM
    balanced_train, validation, test = prepare_data_for_llm(train_df, test_df, prompt_format="feature_enhanced")
    
if __name__ == "__main__":
    main() 