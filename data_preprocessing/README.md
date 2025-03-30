# Data Preprocessing Documentation

## Data Explorer

The `data_explorer.py` script performs comprehensive exploratory data analysis on customer service conversations to understand sentiment patterns and identify key features that influence customer sentiment. This analysis provides critical insights for preparing data for sentiment classification models.

### Purpose

- Analyze the structure and characteristics of customer service conversations
- Identify patterns and relationships between conversation features and sentiment labels
- Extract distinctive linguistic markers for different sentiment categories
- Generate visualizations and statistics to support data-driven decisions

### Recent Updates

The codebase has received the following improvements:

1. **Code Standardization**: All code comments, documentation strings, and variable names have been standardized to English for consistency across the project.

2. **Enhanced Text Preprocessing**: The `preprocess_for_word_analysis` function has been optimized and moved to the global scope for better reusability across both main exploration and LLM-specific exploration functions.

3. **Improved Integration**: A bridge function `get_exploration_findings()` has been added to facilitate seamless integration between the data exploration pipeline and the LLM preprocessing pipeline. This function:
   - Ensures complete exploration is run when needed
   - Creates appropriate output directories for LLM exploration
   - Copies key analysis results between directories for consistency
   - Returns processed data and findings for use in LLM fine-tuning

4. **N-gram and Word Analysis**: The n-gram generation and word analysis components have been refined to produce more meaningful and insightful outputs through better stop word filtering and content-focused processing.

These updates ensure that the data exploration process is more modular, consistent, and produces high-quality analytical outputs for sentiment classification model development.

## LLM Data Preparation

The `preprocess_for_llm.py` script is designed to prepare data for Large Language Model (LLM) fine-tuning for sentiment analysis in customer service conversations. This script performs the following steps:

### 1. Data Loading

```python
def load_data():
    """Load raw data"""
    train_df = pd.read_csv('data/original/train.csv')
    test_df = pd.read_csv('data/original/test.csv')
    
    return train_df, test_df
```

This function loads raw training and test data from the `data/original/` directory. The data contains customer service conversations and their associated sentiment labels.

### 2. Text Preprocessing

#### 2.1 Text Normalization and Cleaning

The `preprocess_text()` function is used to standardize and clean text:
- Converts all text to lowercase
- Replaces URLs with `[URL]` and emails with `[EMAIL]` special tokens
- Replaces numbers and order numbers with the `[NUMBER]` token
- Detects and normalizes patterns like "order number is bb789012"
- Normalizes whitespace and removes unnecessary punctuation
- Includes enhanced regex patterns for alphanumeric order numbers

#### 2.2 Conversation Formatting

The `format_conversation()` function transforms raw conversation data into a format suitable for LLM training:
- Marks customer lines with the `[CUSTOMER]` token
- Marks agent lines with the `[AGENT]` token
- Classifies unmarked lines as customer or agent based on content analysis
- Ensures each conversation has at least one customer and one agent line
- Automatically adds missing speakers when needed

#### 2.3 Customer Text Extraction

The `extract_customer_text()` function extracts only text written by the customer from conversations:
- Identifies lines starting with "Customer:"
- Aggregates customer lines and their continuations
- Creates a more focused text corpus for analyzing customer sentiment

### 3. Prompt Templates and Creation

#### 3.1 Prompt Templates

The `create_prompt_template()` function generates various prompt templates for different LLM training scenarios:
- "full": Basic template with just conversation and sentiment
- "conversation_only": Simple template focused only on conversation
- "customer_only": Template containing only customer messages
- "few_shot": Template containing example scenarios for few-shot learning
- "feature_enhanced": Advanced template incorporating additional features like conversation length, question mark count, etc.

#### 3.2 Prompt Creation

The `create_prompt()` function creates an LLM training prompt for each data row using the specified template:
- Fills in placeholders in the template with actual data
- Adds conversation, customer text, sentiment label, and other features to the prompt
- Calculates additional metrics for feature-enhanced prompts like conversation length, question mark count, etc.

#### 3.3 Prompt Validation

The `validate_prompt()` function checks that generated prompts are correctly formatted for LLM training:
- Verifies the presence of the "Conversation:" section
- Checks that the correct sentiment label is present
- Detects unprocessed order numbers and issues warnings
- Ensures prompt quality through sample validation

### 4. Strategic Sampling and Class Balance

The `calculate_sample_weights()` function implements strategic weighting to address class imbalance in the dataset:
- Assigns weights to each sentiment class inversely proportional to its frequency
- Gives additional weight to examples significantly different from the mean (20% increase if z-score > 1.5)
- Provides additional weight to longer customer messages (10% increase if >100 characters)
- Applies a 50% additional weight boost to positive sentiment examples (which are very rare)

### 5. Dataset Preparation and Splitting

The `prepare_data_for_llm()` function combines all steps in the data preparation process:

1. **Format and Preprocess Conversations**:
   - Format and structure conversations
   - Clean through preprocessing
   - Extract and process customer texts

2. **Feature Engineering**:
   - Conversation length and customer text length
   - Question and exclamation mark counts
   - Customer and agent turn counts

3. **Prompt Creation and Validation**:
   - Template selection and creation
   - Prompt generation for all data rows
   - Sample prompt validation

4. **Strategic Sampling**:
   - Weight calculation
   - Stratified data splitting (StratifiedKFold)
   - Data leakage prevention and control

5. **Balanced Sampling**:
   - Equal sample selection for each sentiment class
   - Weighted random sampling for underrepresented classes
   - Selection of most important examples from well-represented classes

6. **Dataset Saving**:
   - Balanced training set
   - Validation set
   - Enriched test set (original content preserved)
   - Example prompt file

### 6. Main Workflow

The `main()` function organizes the steps of loading data and preparing it for LLM training:
1. Loads raw data
2. Prepares data using the "feature_enhanced" prompt format
3. Creates balanced train, validation, and test sets

## Output Files

When `preprocess_for_llm.py` is executed, it creates the following files in the `data/subdata` directory:

1. **train.csv**: Balanced, strategically sampled training set with class imbalance addressed
2. **validation.csv**: Validation set reserved for model evaluation
3. **test.csv**: Test set with original test data preserved and enhanced with additional features
4. **prompt_examples.txt**: Text file containing example LLM training prompts for each sentiment class

Each dataset contains:
- Original columns: issue_area, product_category, conversation, customer_sentiment, etc.
- Added features:
  - `formatted_conversation`: Conversation transformed into a special format for LLM
  - `customer_text`: Text containing only customer conversations
  - `conversation_length`: Conversation length feature
  - `customer_question_marks`: Question mark count in customer text
  - `customer_turns`: Number of customer turns
  - `prompt`: Complete prompt for LLM training, containing the task and features

## Data Integrity and Protection Measures

The `preprocess_for_llm.py` script implements the following protection measures during data preparation:

1. **Test Dataset Protection**: 
   - Test data never leaks into training or validation data
   - Basic information in the test dataset is preserved unchanged
   - Added columns contain only features necessary for analysis and LLM training

2. **Stratified Splitting**:
   - Ensures sentiment classes are proportionally represented in each dataset
   - Creates reliable and reproducible splits using StratifiedKFold

3. **Balanced Class Distribution**:
   - Provides equal number of examples for each sentiment class (positive, negative, neutral) in the training set
   - Applies weighted random sampling for underrepresented classes, with a 50% additional weight increase specifically for positive examples

4. **Order Number Protection**:
   - Replaces order numbers and other sensitive alphanumeric information with special tokens
   - Detects complex alphanumeric patterns like "order number is bb789012"

5. **Conversation Integrity Preservation**:
   - Ensures both customer and agent contributions are present in conversations
   - Intelligently classifies unmarked lines as customer or agent based on content

These processes ensure that raw data from customer service conversations is reliably transformed into optimized, balanced, and structured datasets for LLM training.

### Methodology

The data exploration script performs a multi-layered analysis:

1. **Basic Data Exploration**:
   - Dataset statistics and distributions
   - Missing value analysis
   - Column information extraction

2. **Sentiment Distribution Analysis**:
   - Overall sentiment counts and percentages
   - Visualization of class imbalance

3. **Customer-Agent Text Analysis**:
   - Separation of customer and agent contributions
   - Text length statistics by sentiment class
   - Word and sentence count analysis
   - Question and exclamation mark usage patterns

4. **Categorical Variable Analysis**:
   - Relationships between issue areas/categories and sentiment
   - Product category sentiment associations
   - Issue complexity impact on sentiment
   - Agent experience level influence

5. **Conversation Flow Analysis**:
   - Turn-taking patterns by sentiment
   - Who starts/ends conversations
   - Average turn lengths
   - Customer-to-agent ratio analysis

6. **Advanced Text Analysis**:
   - Word frequency distributions
   - Word clouds by sentiment
   - TF-IDF analysis for distinctive words
   - N-gram analysis (bigrams/trigrams)

7. **Sentiment Change Analysis**:
   - Tracking sentiment evolution within conversations
   - Initial vs. final sentiment comparisons
   - Sentiment volatility metrics
   - Trend categorization (improving, stable, worsening)

8. **Feature Importance Analysis**:
   - Correlation analysis between features
   - Random Forest-based feature importance ranking
   - Identification of key predictors for sentiment

### Detailed Output Files

The output files produced by the `data_explorer.py` script are organized by analysis type in the `data_preprocessing/data_exploration` directory and include:
- Metadata files: Column information, missing values, statistical summaries
- Distribution analyses: Sentiment distribution, category relationships
- Text statistics: Text lengths by sentiment, punctuation usage
- Word analyses: Frequent words, distinctive words, n-grams
- Conversation flow metrics: Turn counts, start/end patterns
- Sentiment change analysis: Sentiment trends, volatility

### Key Findings and Implications

The most important findings from the analysis:

1. **Class Imbalance**: There is significant imbalance in sentiment classes (55.2% neutral, 42.1% negative, 2.7% positive)

2. **Conversation Length**: There is a strong relationship between sentiment and conversation length (negative: 2398, neutral: 1948, positive: 1682 characters)

3. **Conversation Dynamics**: Negative conversations show distinct interaction patterns (more turns, higher customer-to-agent ratio)

4. **Distinctive Language**:
   - Negative sentiment: Characterized by problem-oriented terms like "refund", "urgently", "shipping"
   - Neutral sentiment: Associated with informational terms like "warranty", "account", "email"
   - Positive sentiment: Contains progress-related terms like "confirm", "status", "information"

5. **Sentiment Evolution**: 62% of negative conversations show improving sentiment, suggesting many initially negative interactions are successfully resolved

These insights inform data preparation strategies (addressing imbalance, feature engineering) and sentiment classification model selection. 