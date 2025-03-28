# Data Preprocessing Documentation

## Data Explorer

The `data_explorer.py` script performs comprehensive exploratory data analysis on customer service conversations to understand sentiment patterns and identify key features that influence customer sentiment. This analysis provides critical insights for preparing data for sentiment classification models.

### Purpose

- Analyze the structure and characteristics of customer service conversations
- Identify patterns and relationships between conversation features and sentiment labels
- Extract distinctive linguistic markers for different sentiment categories
- Generate visualizations and statistics to support data-driven decisions

### Methodology

The script performs a multi-layered analysis:

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

The script generates a comprehensive set of outputs in the `data_preprocessing/data_exploration` directory, organized by analysis type:

#### Metadata Files

- `column_info.json`: 
  Contains detailed information about all columns in the dataset including their names and data types. This file helps understand the structure of customer service conversation data and available features. It lists 11 columns including issue_area, customer_sentiment, product_category, conversation, etc.

- `missing_values.json`: 
  Documents the count of missing values for each column and the total missing values across the dataset. This analysis helps assess data quality and completeness. The dataset shows excellent data quality with no missing values across all columns.

- `stats_summary.csv`: 
  Provides comprehensive statistical measures (mean, std, min, max, quartiles) for all numeric fields and category counts for categorical fields. This summary gives an overview of the distribution characteristics of each variable.

- `analysis_metadata.json`: 
  Contains summary information about the entire analysis process, including dataset size (1000 conversations), sentiment distribution, average lengths, and distinctive words by sentiment. This serves as a quick reference for key dataset properties.

#### Distribution Analysis

- `sentiment_distribution.json`: 
  Contains both raw counts and percentage distribution of sentiment classes. The dataset shows significant imbalance: 55.2% neutral (552), 42.1% negative (421), and only 2.7% positive (27) sentiments.

- `sentiment_distribution.png`: 
  Bar chart visualization of sentiment class distribution, clearly depicting the class imbalance. This visualization helps in understanding the need for balancing strategies during model training.

#### Text Statistics

- `text_stats_by_sentiment.csv`: 
  Detailed breakdown of text metrics for each sentiment category, including:
  - Average conversation length (negative: 2398, neutral: 1948, positive: 1682 characters)
  - Customer text length (negative > neutral > positive)
  - Agent text length by sentiment
  - Word and sentence counts by sentiment
  - Question and exclamation mark usage patterns
  
  This file reveals that negative sentiment conversations are typically longer and contain more customer text than other sentiment categories.

- `text_length_distributions.png`: 
  Four-panel visualization showing box plots of text length distributions by sentiment. The plots display customer text length, agent text length, customer word count, and agent word count across sentiment categories. The visualizations highlight that negative sentiment conversations generally have longer texts and higher word counts.

- `punctuation_by_sentiment.png`: 
  Two-panel bar chart showing question mark and exclamation mark usage by sentiment category. This visualization reveals that customers in negative conversations use more question marks, while positive conversations show higher exclamation mark usage.

#### Categorical Relationships

- `issue_area_sentiment_relationship.csv`, `issue_category_sentiment_relationship.csv`, etc.: 
  Cross-tabulation tables showing the percentage distribution of sentiments across different categorical variables. Each row represents a category (e.g., a specific issue area), and each column shows the percentage of conversations in that category falling into each sentiment class.

- `issue_area_sentiment_heatmap.png`, `issue_category_sentiment_heatmap.png`, etc.: 
  Heatmap visualizations of the relationship between categorical variables and sentiment classes. The intensity of color indicates the percentage of conversations in each category-sentiment combination. These visualizations quickly identify which categories are associated with higher percentages of negative or positive sentiment.

- `category_sentiment_relations.json`: 
  Summary file that identifies the top three categories with the highest negative sentiment percentage for each categorical variable. This information is valuable for identifying problematic areas that may need more attention or improvement.

#### Conversation Flow Analysis

- `flow_metrics_by_sentiment.csv`: 
  Comprehensive metrics on conversation dynamics by sentiment category:
  - Customer turns (negative: 8.06, neutral: 7.43, positive: 6.78)
  - Agent turns (negative: 7.53, neutral: 7.05, positive: 6.78)
  - Total turns (negative > neutral > positive)
  - Turn ratio (customer turns/agent turns)
  - Average customer and agent turn lengths
  
  This analysis shows that negative conversations typically involve more turns and back-and-forth interactions.

- `conversation_flow_metrics.png`: 
  Three-panel visualization showing:
  1. Average number of turns by sentiment
  2. Customer to agent turn ratio by sentiment
  3. Average turn length comparison between customer and agent by sentiment
  
  These visualizations highlight that negative sentiment conversations have more turns and higher customer-to-agent turn ratios.

- `conversation_start_end_patterns.png`: 
  Two-panel visualization showing:
  1. Who started the conversation (customer vs. agent) by sentiment category
  2. Who ended the conversation by sentiment category
  
  This analysis reveals patterns in conversation initiation and conclusion across sentiment classes.

- `sentiment_by_length_category.csv`: 
  Analysis of sentiment distribution across different conversation length categories (short, medium, long). The results show that 100% of short conversations (<500 characters) are negative, 100% of medium conversations are neutral, and long conversations have a mix of sentiments (42.1% negative, 55.1% neutral, 2.7% positive).

- `sentiment_by_turn_category.csv`: 
  Analysis of sentiment distribution across different turn count categories (few, medium, many). Shows that 80% of conversations with few turns are negative, while conversations with medium or many turns have a more balanced sentiment distribution.

- `sentiment_by_conversation_structure.png`: 
  Two-panel visualization showing:
  1. Stacked bar chart of sentiment distribution by conversation length
  2. Stacked bar chart of sentiment distribution by turn count
  
  These visualizations highlight the strong relationship between conversation structure and sentiment.

#### Word Analysis

- `customer_words_by_sentiment.json`: 
  Contains the top 20 most frequent words used by customers for each sentiment category, with their frequency counts. This analysis helps identify common vocabulary patterns across sentiment classes. For example:
  - Neutral: "thank", "help", "okay", "yes", "number", "email", etc.
  - Negative: "thank", "order", "yes", "okay", "help", "number", etc.
  - Positive: "thank", "order", "help", "yes", "great", etc.

- `top_customer_words.png`: 
  Bar chart visualization of the 20 most frequent customer words across all sentiments. This chart provides an overview of common vocabulary in customer communications.

- `sentiment_wordclouds.png`: 
  Three-panel visualization showing word clouds for each sentiment category. The size of each word represents its frequency within that sentiment class. This visually striking representation makes it easy to identify dominant vocabulary patterns for each sentiment.

- `tfidf_distinctive_words.json`: 
  Contains the top 15 most distinctive words for each sentiment category as identified by TF-IDF analysis. Unlike simple word frequency, this analysis finds words that are uniquely characteristic of each sentiment class:
  - Negative: "refund", "urgently", "shipping", "delivered", etc.
  - Neutral: "warranty", "mobile", "account", "email", etc.
  - Positive: "status", "confirm", "information", "placed", etc.

- `tfidf_distinctive_words.png`: 
  Three-panel horizontal bar chart showing the distinctiveness score of the top 10 words for each sentiment category. The length of each bar represents how strongly that word distinguishes a particular sentiment from others.

- `ngram_analysis.json`: 
  Contains the top 10 most common bigrams (two-word phrases) and trigrams (three-word phrases) for each sentiment category. This analysis reveals common multi-word expressions associated with each sentiment:
  - Negative bigrams: "order number", "thank help", "email address", etc.
  - Neutral bigrams: "thank help", "order number", "email address", etc.
  - Positive bigrams: "thank help", "order status", "order number", etc.

#### Sentiment Change Analysis

- `sentiment_change_metrics.csv`: 
  Contains metrics tracking how sentiment evolves within conversations for each sentiment category:
  - Customer initial sentiment (compound score at the start)
  - Customer final sentiment (compound score at the end)
  - Agent initial and final sentiment
  - Sentiment volatility (standard deviation of sentiment throughout conversation)
  - Average sentiment change
  - Customer sentiment shift (final - initial)
  
  This analysis shows that negative conversations typically show the largest positive shift in sentiment (+0.38), suggesting effective agent intervention.

- `sentiment_trend_distribution.csv`: 
  Detailed breakdown of sentiment trend patterns (improving, stable, worsening) by final sentiment category. For example, of the 421 negative sentiment conversations:
  - 262 (62.2%) show improving sentiment
  - 112 (26.6%) remain stable
  - 44 (10.5%) show worsening sentiment
  
  This distribution indicates that most negative conversations actually show improvement in customer sentiment by the end.

- `sentiment_change_analysis.png`: 
  Three-panel visualization showing:
  1. Average sentiment shift by category
  2. Sentiment volatility by category
  3. Comparison of initial vs. final sentiment scores
  
  These visualizations highlight how sentiment typically changes during conversations with different final sentiment classifications.

- `sentiment_trend_distribution.png`: 
  Stacked bar chart showing the percentage distribution of sentiment trends (improving, stable, worsening) for each sentiment category. This visualization makes it easy to compare trend patterns across sentiment classes.

#### Feature Analysis

- `correlation_matrix.csv`: 
  Full correlation matrix between all numeric features (17x17). This matrix quantifies the relationship strength between pairs of features, with values ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).

- `correlation_heatmap.png`: 
  Heatmap visualization of the correlation matrix, with color intensity representing correlation strength. This visualization highlights feature relationships such as:
  - Strong positive correlation between conversation length and word counts
  - Moderate positive correlation between customer and agent text lengths
  - Noteworthy correlations between sentiment volatility and conversation structure metrics

- `feature_importance.csv`: 
  Ranked list of features by their importance for sentiment prediction, as determined by a Random Forest classifier. The top features include:
  1. Sentiment volatility (16.3%)
  2. Conversation length (11.8%)
  3. Agent word count (9.0%)
  4. Agent text length (8.6%)
  5. Customer sentiment shift (7.3%)
  
  This ranking helps identify which features are most predictive of sentiment, guiding feature engineering efforts.

- `feature_importance.png`: 
  Bar chart visualization of feature importance scores. This visualization makes it easy to compare the relative predictive power of different features for sentiment classification.

#### Summary

- `key_findings.txt`: 
  Concise summary of the most important discoveries from all analyses, including:
  1. Dataset statistics and class distribution
  2. Sentiment-specific conversation characteristics
  3. Most important features for sentiment prediction
  4. Distinctive words for each sentiment category
  
  This file serves as a quick reference for the most actionable insights from the exploratory analysis.

### Key Findings and Implications

The analysis revealed several important patterns with direct implications for sentiment classification modeling:

1. **Class Imbalance**: The significant imbalance in sentiment classes (55.2% neutral, 42.1% negative, 2.7% positive) suggests that sampling strategies such as oversampling positive examples or balanced batching during training may be necessary.

2. **Conversation Length**: The strong relationship between sentiment and conversation length (negative: 2398, neutral: 1948, positive: 1682 characters) indicates that text length features should be included in the model.

3. **Turn Dynamics**: Negative conversations have distinctly different interaction patterns (more turns, higher customer-to-agent ratio), suggesting that conversation flow features are important indicators of sentiment.

4. **Conversation Structure**: The perfect correlation between conversation length categories and sentiment (short → negative, medium → neutral) highlights the predictive power of structural features.

5. **Distinctive Language**: 
   - Negative sentiment is characterized by problem-oriented terms like "refund," "urgently," "shipping"
   - Neutral sentiment is associated with informational terms like "warranty," "account," "email"
   - Positive sentiment includes progress-related terms like "confirm," "status," "information"
   
   These linguistic markers support the use of bag-of-words or embedding-based features.

6. **Sentiment Evolution**: The finding that 62% of negative conversations show improving sentiment suggests that tracking sentiment change could be a valuable feature. It also indicates that many initially negative interactions are successfully resolved.

7. **Feature Importance**: The dominance of sentiment volatility, conversation length, and agent word count as predictive features suggests a combined approach that incorporates both structural and content-based features.

These insights inform both preprocessing strategies (handling imbalance, feature engineering) and model selection (models capable of capturing both text content and conversation structure) for sentiment classification.

### Using the Analysis Results

To leverage these insights for model development:

1. **Feature Engineering**: Create features based on the most predictive characteristics identified:
   - Conversation structure features (length, turns, ratio)
   - Sentiment change features (initial sentiment, volatility)
   - Text-based features (distinctive words, n-grams)

2. **Data Balancing**: Implement strategies to address class imbalance:
   - Oversampling positive examples
   - Undersampling neutral examples
   - Class weighting during training

3. **Model Selection**: Choose models that can capture both:
   - Text content (transformer-based models like BERT, RoBERTa)
   - Conversation dynamics (sequential models, possibly with features from this analysis)

4. **Evaluation Strategy**: When evaluating models:
   - Use balanced metrics (F1 score, balanced accuracy)
   - Pay special attention to performance on minority classes
   - Consider separate evaluation on different conversation length categories

By incorporating these insights, sentiment classification models can be better designed to capture the nuanced patterns in customer service conversations.
