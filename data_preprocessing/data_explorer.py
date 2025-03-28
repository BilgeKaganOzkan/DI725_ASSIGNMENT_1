import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
import json
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'data_preprocessing/data_exploration'
os.makedirs(output_dir, exist_ok=True)

# Basic utility functions
def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_to_json(data, filename):
    """Save data to JSON file with nice formatting."""
    # Convert numpy types to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        else:
            return obj
    
    converted_data = convert_numpy_types(data)
    with open(f"{output_dir}/{filename}", 'w') as f:
        json.dump(converted_data, f, indent=4)

def save_to_txt(text, filename):
    """Save text to a file."""
    with open(f"{output_dir}/{filename}", 'w') as f:
        f.write(text)

def save_to_csv(df, filename):
    """Save dataframe to CSV."""
    df.to_csv(f"{output_dir}/{filename}")

# Text processing functions
def extract_customer_text(conversation):
    """Extract only customer parts from the conversation."""
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
            # Remove "Customer:" label
            customer_text = line[9:].strip()
            if customer_text:  # If line is not empty
                customer_lines.append(customer_text)
        elif line.startswith('Agent:'):
            is_customer = False
        elif is_customer and line:  # If still customer talking and line is not empty
            customer_lines.append(line)
    
    return ' '.join(customer_lines)

def extract_agent_text(conversation):
    """Extract only agent parts from the conversation."""
    if not isinstance(conversation, str):
        return ""
    
    lines = conversation.split('\n')
    agent_lines = []
    is_agent = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Agent:'):
            is_agent = True
            # Remove "Agent:" label
            agent_text = line[6:].strip()
            if agent_text:  # If line is not empty
                agent_lines.append(agent_text)
        elif line.startswith('Customer:'):
            is_agent = False
        elif is_agent and line:  # If still agent talking and line is not empty
            agent_lines.append(line)
    
    return ' '.join(agent_lines)

def count_sentences(text):
    """Count sentences in text using a more robust method."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(sent_tokenize(text))

def preprocess_text(text, remove_stopwords=False):
    """Apply text preprocessing with options."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stop words if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Final cleanup of extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def count_question_marks(text):
    """Count question marks in text."""
    if not isinstance(text, str):
        return 0
    return text.count('?')

def count_exclamation_marks(text):
    """Count exclamation marks in text."""
    if not isinstance(text, str):
        return 0
    return text.count('!')

def get_ngrams(text, n=2):
    """Extract meaningful n-grams from text.
    Focuses on content-rich phrases by removing stop words."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Basic preprocessing to standardize text
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize using NLTK for better handling
    tokens = word_tokenize(text)
    
    # Define specific stop words to remove for n-grams
    # For n-grams, we want to be more selective about what we remove
    # to retain meaningful phrases
    stop_words = set(stopwords.words('english'))
    
    # Common words that don't add meaning in n-grams
    ngram_stop_words = {
        'would', 'could', 'should', 'please', 'thank', 'thanks',
        'hello', 'hi', 'hey', 'okay', 'ok', 'yes', 'no', 'yeah',
        'bye', 'goodbye', 'welcome', 'appreciated', 'got', 'get',
        'will', 'shall', 'all', 'ive', 'im', 'thats', 'its',
        'just', 'also', 'right', 'actually', 'really', 'use'
    }
    
    # Use a smaller set of stop words to preserve more phrase structure
    filtered_stop_words = ngram_stop_words.union(
        {word for word in stop_words if len(word) <= 3}
    )
    
    # Filter out only common stop words, keeping more content words
    filtered_tokens = [
        token for token in tokens 
        if token not in filtered_stop_words and len(token) > 2 and token.isalpha()
    ]
    
    # Generate n-grams only from meaningful tokens
    if len(filtered_tokens) < n:
        return []
    
    n_grams = list(ngrams(filtered_tokens, n))
    
    # Return joined n-grams
    return [' '.join(g) for g in n_grams]

# Analysis functions
def analyze_conversation_flow(conversation):
    """Analyze the flow of conversation between customer and agent."""
    if not isinstance(conversation, str):
        return {
            "started_by": "Unknown",
            "ended_by": "Unknown",
            "customer_turns": 0,
            "agent_turns": 0,
            "total_turns": 0,
            "turn_ratio": 0.0,
            "avg_customer_turn_length": 0,
            "avg_agent_turn_length": 0
        }
    
    lines = conversation.split('\n')
    turns = []
    current_speaker = None
    customer_turn_lengths = []
    agent_turn_lengths = []
    current_turn_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Customer:'):
            if current_speaker == 'Agent' and current_turn_text:
                agent_turn_lengths.append(len(current_turn_text))
                current_turn_text = ""
                
            if current_speaker != 'Customer':
                turns.append('Customer')
                current_speaker = 'Customer'
                current_turn_text = line[9:].strip()  # Remove "Customer:" prefix
            else:
                current_turn_text += " " + line
                
        elif line.startswith('Agent:'):
            if current_speaker == 'Customer' and current_turn_text:
                customer_turn_lengths.append(len(current_turn_text))
                current_turn_text = ""
                
            if current_speaker != 'Agent':
                turns.append('Agent')
                current_speaker = 'Agent'
                current_turn_text = line[6:].strip()  # Remove "Agent:" prefix
            else:
                current_turn_text += " " + line
    
    # Don't forget the last turn
    if current_speaker == 'Customer' and current_turn_text:
        customer_turn_lengths.append(len(current_turn_text))
    elif current_speaker == 'Agent' and current_turn_text:
        agent_turn_lengths.append(len(current_turn_text))
    
    # Calculate metrics
    customer_turns = turns.count('Customer')
    agent_turns = turns.count('Agent')
    
    # Check who started and ended the conversation
    started_by = turns[0] if turns else "Unknown"
    ended_by = turns[-1] if turns else "Unknown"
    
    # Calculate average turn lengths
    avg_customer_turn_length = np.mean(customer_turn_lengths) if customer_turn_lengths else 0
    avg_agent_turn_length = np.mean(agent_turn_lengths) if agent_turn_lengths else 0
    
    return {
        "started_by": started_by,
        "ended_by": ended_by,
        "customer_turns": customer_turns,
        "agent_turns": agent_turns,
        "total_turns": customer_turns + agent_turns,
        "turn_ratio": customer_turns / max(1, agent_turns),  # Avoid division by zero
        "avg_customer_turn_length": avg_customer_turn_length,
        "avg_agent_turn_length": avg_agent_turn_length
    }

def analyze_sentiment_changes(conversation):
    """Analyze sentiment changes throughout conversation."""
    sia = SentimentIntensityAnalyzer()
    
    if not isinstance(conversation, str):
        return {
            "customer_initial_sentiment": 0,
            "customer_final_sentiment": 0,
            "agent_initial_sentiment": 0,
            "agent_final_sentiment": 0,
            "sentiment_volatility": 0,
            "avg_sentiment_change": 0,
            "sentiment_trend": "neutral"
        }
    
    # Split conversation into customer and agent parts
    lines = conversation.split('\n')
    customer_parts = []
    agent_parts = []
    current_speaker = None
    current_part = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Customer:'):
            if current_speaker == 'Agent' and current_part:
                agent_parts.append(current_part)
                current_part = ""
            
            current_speaker = 'Customer'
            current_part = line[9:].strip() if current_part == "" else current_part + " " + line[9:].strip()
            
        elif line.startswith('Agent:'):
            if current_speaker == 'Customer' and current_part:
                customer_parts.append(current_part)
                current_part = ""
            
            current_speaker = 'Agent'
            current_part = line[6:].strip() if current_part == "" else current_part + " " + line[6:].strip()
            
        else:
            current_part += " " + line
    
    # Add the last part
    if current_speaker == 'Customer' and current_part:
        customer_parts.append(current_part)
    elif current_speaker == 'Agent' and current_part:
        agent_parts.append(current_part)
    
    # Calculate sentiments
    customer_sentiments = [sia.polarity_scores(part)['compound'] for part in customer_parts]
    agent_sentiments = [sia.polarity_scores(part)['compound'] for part in agent_parts]
    
    # Calculate sentiment changes
    sentiment_changes = []
    for i in range(len(customer_sentiments) - 1):
        sentiment_changes.append(customer_sentiments[i+1] - customer_sentiments[i])
    
    # Determine sentiment trend
    if not customer_sentiments:
        sentiment_trend = "neutral"
    elif len(customer_sentiments) < 2:
        sentiment_trend = "stable"
    else:
        initial = customer_sentiments[0]
        final = customer_sentiments[-1]
        if final - initial > 0.2:
            sentiment_trend = "improving"
        elif final - initial < -0.2:
            sentiment_trend = "worsening"
        else:
            sentiment_trend = "stable"
    
    return {
        "customer_initial_sentiment": customer_sentiments[0] if customer_sentiments else 0,
        "customer_final_sentiment": customer_sentiments[-1] if customer_sentiments else 0,
        "agent_initial_sentiment": agent_sentiments[0] if agent_sentiments else 0,
        "agent_final_sentiment": agent_sentiments[-1] if agent_sentiments else 0,
        "sentiment_volatility": np.std(customer_sentiments) if customer_sentiments else 0,
        "avg_sentiment_change": np.mean(sentiment_changes) if sentiment_changes else 0,
        "sentiment_trend": sentiment_trend
    }

def compute_tf_idf_by_sentiment(df, text_column, sentiment_column):
    """Compute TF-IDF scores to find distinctive words for each sentiment class.
    Enhanced to find truly distinctive words with higher discriminative power."""
    
    # Comprehensive set of stop words
    stop_words = set(stopwords.words('english'))
    customer_service_common = {
        # Greetings and formalities
        'would', 'could', 'should', 'please', 'thank', 'thanks', 'hello', 'hi', 'hey',
        'okay', 'ok', 'yes', 'yep', 'yeah', 'sure', 'well', 'good', 'great', 'welcome',
        'goodbye', 'bye', 'welcome', 'appreciate', 'assist', 'assistance', 'helping',
        
        # Common verbs without context
        'got', 'get', 'getting', 'let', 'going', 'go', 'come', 'see', 'know', 'need',
        'want', 'make', 'take', 'put', 'ask', 'tell', 'told', 'said', 'says',
        
        # Pronouns and names
        'sir', 'maam', 'madam', 'ms', 'mr', 'miss', 'dear', 'name',
        
        # Time expressions
        'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there',
        
        # Question words
        'where', 'how', 'what', 'who', 'why', 'when',
        
        # Common customer service verbs
        'call', 'calling', 'called', 'say', 'said', 'says', 'solve', 'resolved',
        
        # Other fillers
        'also', 'alright', 'actually', 'basically', 'definitely', 'absolutely'
    }
    all_stop_words = stop_words.union(customer_service_common)
    
    sentiment_groups = {}
    sentiment_differential_words = {}
    
    # First run: basic TF-IDF to find candidate words
    for sentiment in df[sentiment_column].unique():
        sentiment_docs = df[df[sentiment_column] == sentiment][text_column]
        
        if len(sentiment_docs) > 0 and sentiment_docs.str.strip().str.len().sum() > 0:
            try:
                # More specific TF-IDF settings
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=100,  # Increase to find more candidates
                    stop_words=list(all_stop_words),
                    min_df=2,  # Appear in at least 2 documents
                    max_df=0.7,  # More restrictive upper bound (was 0.8)
                    ngram_range=(1, 1)  # Unigrams only
                )
                
                sentiment_tfidf = tfidf_vectorizer.fit_transform(sentiment_docs)
                sentiment_feature_names = tfidf_vectorizer.get_feature_names_out()
                
                # Store results
                sentiment_groups[sentiment] = {
                    'docs': sentiment_docs,
                    'features': sentiment_feature_names,
                    'tfidf': sentiment_tfidf,
                    'vectorizer': tfidf_vectorizer
                }
            except ValueError as e:
                print(f"Error processing sentiment '{sentiment}': {e}")
                sentiment_groups[sentiment] = {}
        else:
            sentiment_groups[sentiment] = {}
    
    # Second run: Calculate differential TF-IDF with higher contrast
    for target_sentiment, group_data in sentiment_groups.items():
        if not group_data:
            sentiment_differential_words[target_sentiment] = {}
            continue
            
        distinctive_words = {}
        
        for word in group_data['features']:
            word_idx = np.where(group_data['features'] == word)[0]
            if len(word_idx) > 0:
                target_score = group_data['tfidf'][:, word_idx[0]].mean()
                
                # Compare with other sentiments
                other_scores = []
                for other_sentiment, other_data in sentiment_groups.items():
                    if other_sentiment != target_sentiment and other_data:
                        other_word_idx = np.where(other_data['features'] == word)[0]
                        if len(other_word_idx) > 0:
                            other_score = other_data['tfidf'][:, other_word_idx[0]].mean()
                            other_scores.append(other_score)
                        else:
                            other_scores.append(0)
                
                # Calculate distinctiveness with higher contrast
                if other_scores:
                    avg_other_score = sum(other_scores) / len(other_scores)
                    # Apply a more aggressive differential 
                    distinctiveness = float(target_score) - (avg_other_score * 1.5)
                    
                    if distinctiveness > 0:
                        # Check if word length is meaningful
                        if len(word) >= 3:
                            distinctive_words[word] = distinctiveness
        
        # Sort and keep top distinctive words
        sentiment_differential_words[target_sentiment] = dict(
            sorted(distinctive_words.items(), key=lambda x: x[1], reverse=True)[:25]
        )
    
    return sentiment_differential_words

# Let's define the preprocess_for_word_analysis function in global scope first,
# so it can be used in both main() and explore_llm_data() functions.

def preprocess_for_word_analysis(text):
    """Enhanced text preprocessing for word analysis.
    Removes stop words, short words, and common service words.
    Focuses on meaningful content words."""
    # Check for empty or non-string input
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove very short words
    words = [word for word in text.split() if len(word) > 2]
    
    # Define more comprehensive stop words including customer service common terms
    stop_words = set(stopwords.words('english'))
    
    # Common words in customer service conversations that don't convey sentiment or topic
    customer_service_common = {
        # Greetings and formalities
        'would', 'could', 'should', 'please', 'thank', 'thanks', 'hello', 'hi', 'hey',
        'okay', 'ok', 'yes', 'yep', 'yeah', 'sure', 'well', 'good', 'great', 'welcome',
        'goodbye', 'bye', 'welcome', 'appreciate', 'assist', 'assistance', 'helping',
        
        # Common verbs without context
        'got', 'get', 'getting', 'let', 'going', 'go', 'come', 'see', 'know', 'need',
        'want', 'make', 'take', 'put', 'ask', 'tell', 'told', 'said', 'says',
        
        # Pronouns and determiners
        'all', 'ive', 'im', 'thats', 'its', 'this', 'that', 'those', 'these', 'their',
        'our', 'your', 'mine', 'yours', 'ours', 'theirs', 'him', 'her', 'them', 'they',
        
        # Other fillers and non-descriptive words
        'just', 'also', 'right', 'now', 'then', 'there', 'here', 'today', 'yesterday',
        'tomorrow', 'ago', 'sometimes', 'often', 'always', 'never', 'ever', 'actually',
        'really', 'basically', 'simply', 'generally', 'usually'
    }
    
    # Combine all stop words
    all_stop_words = stop_words.union(customer_service_common)
    
    # Keep words that:
    # 1. Are not in stop words
    # 2. Are longer than 2 characters
    filtered_words = [word for word in words if word not in all_stop_words]
    
    return filtered_words

def main():
    """Main function to run the comprehensive data exploration."""
    print("Starting comprehensive data exploration...")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('data/original/train.csv')
    test_df = pd.read_csv('data/original/test.csv')
    
    # Display basic information
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # Check columns in the datasets
    print("\nColumns in the dataset:")
    print(train_df.columns.tolist())
    
    # Save column information
    column_info = {
        "columns": train_df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in train_df.dtypes.items()}
    }
    save_to_json(column_info, "column_info.json")
    
    # Check for missing values
    missing_values = train_df.isnull().sum().to_dict()
    missing_values['total'] = train_df.isnull().sum().sum()
    print("\nMissing values:")
    print(missing_values)
    save_to_json(missing_values, "missing_values.json")
    
    # Statistical summary
    print("\nGenerating statistical summary...")
    stats_summary = train_df.describe(include='all')
    save_to_csv(stats_summary, "stats_summary.csv")
    
    # Create combined dataset for analysis
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Sentiment distribution analysis
    sentiment_dist = all_data['customer_sentiment'].value_counts()
    sentiment_pct = all_data['customer_sentiment'].value_counts(normalize=True) * 100
    
    sentiment_distribution = {
        "counts": sentiment_dist.to_dict(),
        "percentages": sentiment_pct.to_dict()
    }
    print("\nSentiment distribution:")
    print(sentiment_distribution)
    save_to_json(sentiment_distribution, "sentiment_distribution.json")
    
    # Visualization: Sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='customer_sentiment', data=all_data)
    plt.title('Distribution of Sentiment Classes')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/sentiment_distribution.png")
    
    # Add categorical version of sentiment for nicer labels
    all_data['sentiment_cat'] = all_data['customer_sentiment'].map({
        'neutral': 'Neutral', 
        'negative': 'Negative', 
        'positive': 'Positive'
    })
    
    # PART 1: Customer-Agent Text Analysis
    print("\nExtracting customer and agent texts...")
    all_data['customer_text'] = all_data['conversation'].apply(extract_customer_text)
    all_data['agent_text'] = all_data['conversation'].apply(extract_agent_text)
    
    # Text statistics
    all_data['conversation_length'] = all_data['conversation'].str.len()
    all_data['customer_text_length'] = all_data['customer_text'].str.len()
    all_data['agent_text_length'] = all_data['agent_text'].str.len()
    all_data['customer_word_count'] = all_data['customer_text'].apply(lambda x: len(str(x).split()))
    all_data['agent_word_count'] = all_data['agent_text'].apply(lambda x: len(str(x).split()))
    all_data['customer_sentence_count'] = all_data['customer_text'].apply(count_sentences)
    all_data['agent_sentence_count'] = all_data['agent_text'].apply(count_sentences)
    
    # Add question and exclamation mark counts
    all_data['customer_question_marks'] = all_data['customer_text'].apply(count_question_marks)
    all_data['customer_exclamation_marks'] = all_data['customer_text'].apply(count_exclamation_marks)
    all_data['agent_question_marks'] = all_data['agent_text'].apply(count_question_marks)
    all_data['agent_exclamation_marks'] = all_data['agent_text'].apply(count_exclamation_marks)
    
    # Sentiment-specific text statistics
    text_stats_by_sentiment = all_data.groupby('sentiment_cat')[
        ['conversation_length', 'customer_text_length', 'agent_text_length',
         'customer_word_count', 'agent_word_count', 
         'customer_sentence_count', 'agent_sentence_count',
         'customer_question_marks', 'customer_exclamation_marks',
         'agent_question_marks', 'agent_exclamation_marks']
    ].mean().reset_index()
    
    print("\nText statistics by sentiment:")
    print(text_stats_by_sentiment)
    save_to_csv(text_stats_by_sentiment, "text_stats_by_sentiment.csv")
    
    # Visualization: Text length distributions
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='sentiment_cat', y='customer_text_length', data=all_data)
    plt.title('Customer Text Length by Sentiment')
    plt.ylabel('Length (characters)')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='sentiment_cat', y='agent_text_length', data=all_data)
    plt.title('Agent Text Length by Sentiment')
    plt.ylabel('Length (characters)')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='sentiment_cat', y='customer_word_count', data=all_data)
    plt.title('Customer Word Count by Sentiment')
    plt.ylabel('Word Count')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='sentiment_cat', y='agent_word_count', data=all_data)
    plt.title('Agent Word Count by Sentiment')
    plt.ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_distributions.png")
    
    # Visualization: Question and exclamation marks
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='sentiment_cat', y='customer_question_marks', data=all_data)
    plt.title('Customer Question Marks by Sentiment')
    plt.ylabel('Average Count')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='sentiment_cat', y='customer_exclamation_marks', data=all_data)
    plt.title('Customer Exclamation Marks by Sentiment')
    plt.ylabel('Average Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/punctuation_by_sentiment.png")
    
    # PART 2: Categorical Variable Relationships with Sentiment
    print("\nAnalyzing categorical relationships with sentiment...")
    categorical_cols = ['issue_area', 'issue_category', 'product_category', 
                        'issue_complexity', 'agent_experience_level']
    
    category_sentiment_relations = {}
    
    for col in categorical_cols:
        if col in all_data.columns:
            # Cross-tabulation
            cross_tab = pd.crosstab(
                all_data[col], 
                all_data['customer_sentiment'], 
                normalize='index'
            ) * 100
            
            # Save results
            save_to_csv(cross_tab, f"{col}_sentiment_relationship.csv")
            
            # Store top categories with negative sentiment
            top_negative = cross_tab.sort_values('negative', ascending=False).index[:3].tolist()
            category_sentiment_relations[col] = {
                "top_negative_categories": top_negative,
                "relationship_description": f"Top {col} categories with negative sentiment: {', '.join(top_negative)}"
            }
            
            # Visualization: Heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='.1f')
            plt.title(f'Relationship Between {col} and Customer Sentiment (%)')
            plt.xlabel('Customer Sentiment')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_sentiment_heatmap.png")
            
    # Save category-sentiment relationships
    save_to_json(category_sentiment_relations, "category_sentiment_relations.json")
    
    # PART 3: Conversation Flow Analysis
    print("\nAnalyzing conversation flow dynamics...")
    all_data['conversation_flow'] = all_data['conversation'].apply(analyze_conversation_flow)
    
    # Extract flow features
    all_data['started_by'] = all_data['conversation_flow'].apply(lambda x: x['started_by'])
    all_data['ended_by'] = all_data['conversation_flow'].apply(lambda x: x['ended_by'])
    all_data['customer_turns'] = all_data['conversation_flow'].apply(lambda x: x['customer_turns'])
    all_data['agent_turns'] = all_data['conversation_flow'].apply(lambda x: x['agent_turns'])
    all_data['total_turns'] = all_data['conversation_flow'].apply(lambda x: x['total_turns'])
    all_data['turn_ratio'] = all_data['conversation_flow'].apply(lambda x: x['turn_ratio'])
    all_data['avg_customer_turn_length'] = all_data['conversation_flow'].apply(lambda x: x['avg_customer_turn_length'])
    all_data['avg_agent_turn_length'] = all_data['conversation_flow'].apply(lambda x: x['avg_agent_turn_length'])
    
    # Analyze flow metrics by sentiment
    flow_by_sentiment = all_data.groupby('sentiment_cat')[
        ['customer_turns', 'agent_turns', 'total_turns', 'turn_ratio',
         'avg_customer_turn_length', 'avg_agent_turn_length']
    ].mean().reset_index()
    
    print("\nConversation Flow Metrics by Sentiment:")
    print(flow_by_sentiment)
    save_to_csv(flow_by_sentiment, "flow_metrics_by_sentiment.csv")
    
    # Visualization: Conversation turns by sentiment
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.barplot(x='sentiment_cat', y='total_turns', data=flow_by_sentiment, palette='viridis')
    plt.title('Average Number of Turns by Sentiment')
    plt.ylabel('Average Turns')
    
    plt.subplot(1, 3, 2)
    sns.barplot(x='sentiment_cat', y='turn_ratio', data=flow_by_sentiment, palette='viridis')
    plt.title('Customer to Agent Turn Ratio by Sentiment')
    plt.ylabel('Customer/Agent Ratio')
    
    plt.subplot(1, 3, 3)
    
    # Create a melted dataframe for comparison
    turn_lengths = flow_by_sentiment[['sentiment_cat', 'avg_customer_turn_length', 'avg_agent_turn_length']]
    turn_lengths_melted = pd.melt(turn_lengths, 
                                  id_vars=['sentiment_cat'], 
                                  value_vars=['avg_customer_turn_length', 'avg_agent_turn_length'],
                                  var_name='Turn Type',
                                  value_name='Average Length')
    turn_lengths_melted['Turn Type'] = turn_lengths_melted['Turn Type'].map({
        'avg_customer_turn_length': 'Customer',
        'avg_agent_turn_length': 'Agent'
    })
    
    sns.barplot(x='sentiment_cat', y='Average Length', hue='Turn Type', data=turn_lengths_melted, palette='viridis')
    plt.title('Average Turn Length by Sentiment')
    plt.ylabel('Average Characters per Turn')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/conversation_flow_metrics.png")
    
    # Visualization: Conversation starts and ends
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    conversation_starters = pd.crosstab(all_data['sentiment_cat'], all_data['started_by'], normalize='index') * 100
    conversation_starters.plot(kind='bar', stacked=True)
    plt.title('Who Started the Conversation by Sentiment')
    plt.ylabel('Percentage')
    plt.xlabel('Sentiment')
    
    plt.subplot(1, 2, 2)
    conversation_enders = pd.crosstab(all_data['sentiment_cat'], all_data['ended_by'], normalize='index') * 100
    conversation_enders.plot(kind='bar', stacked=True)
    plt.title('Who Ended the Conversation by Sentiment')
    plt.ylabel('Percentage')
    plt.xlabel('Sentiment')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/conversation_start_end_patterns.png")
    
    # Define and analyze conversation length categories
    all_data['conversation_length_category'] = pd.cut(
        all_data['conversation_length'],
        bins=[0, 500, 1000, float('inf')],
        labels=['Short', 'Medium', 'Long']
    )
    
    all_data['turn_count_category'] = pd.cut(
        all_data['total_turns'],
        bins=[0, 4, 8, float('inf')],
        labels=['Few turns', 'Medium turns', 'Many turns']
    )
    
    # Sentiment distribution by conversation length and turns
    length_sentiment = pd.crosstab(
        all_data['conversation_length_category'],
        all_data['sentiment_cat'],
        normalize='index'
    ) * 100
    
    turn_sentiment = pd.crosstab(
        all_data['turn_count_category'],
        all_data['sentiment_cat'],
        normalize='index'
    ) * 100
    
    print("\nSentiment Distribution by Conversation Length (%):")
    print(length_sentiment)
    print("\nSentiment Distribution by Turn Count (%):")
    print(turn_sentiment)
    save_to_csv(length_sentiment, "sentiment_by_length_category.csv")
    save_to_csv(turn_sentiment, "sentiment_by_turn_category.csv")
    
    # Visualize sentiment distribution by length/turn categories
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    length_sentiment.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Conversation Length')
    plt.ylabel('Percentage')
    plt.xlabel('Conversation Length')
    
    plt.subplot(1, 2, 2)
    turn_sentiment.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Turn Count')
    plt.ylabel('Percentage')
    plt.xlabel('Turn Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_by_conversation_structure.png")
    
    # PART 4: Advanced Text Analysis (Word Frequency, N-grams, TF-IDF)
    print("\nPerforming advanced text analysis...")
    # Preprocess text for analysis
    all_data['preprocessed_customer_text'] = all_data['customer_text'].apply(
        lambda x: preprocess_text(str(x), remove_stopwords=True)
    )
    
    all_data['preprocessed_agent_text'] = all_data['agent_text'].apply(
        lambda x: preprocess_text(str(x), remove_stopwords=True)
    )
    
    # Most common words in conversations by sentiment - improved with better filtering
    customer_words_by_sentiment = {}
    for sentiment in all_data['customer_sentiment'].unique():
        # Get all customer texts for this sentiment
        sentiment_texts = all_data[all_data['customer_sentiment'] == sentiment]['customer_text']
        
        # Apply enhanced text preprocessing
        all_words = []
        for text in sentiment_texts:
            # Use the same preprocessing as in explore_llm_data for consistency
            words = preprocess_for_word_analysis(text)
            all_words.extend(words)
            
        # Get most common words
        word_counts = Counter(all_words)
        customer_words_by_sentiment[sentiment] = dict(word_counts.most_common(20))
    
    # Save word frequency data in improved format
    save_to_json(customer_words_by_sentiment, "customer_words_by_sentiment.json")
    
    # All customer words for overall analysis
    all_customer_words = []
    for text in all_data['customer_text']:
        all_customer_words.extend(preprocess_for_word_analysis(text))
    
    top_customer_words = dict(Counter(all_customer_words).most_common(30))
    
    print("\nTop 20 customer words overall:")
    print(list(top_customer_words.keys())[:20])
    
    print("\nTop 10 customer words by sentiment:")
    for sentiment, words in customer_words_by_sentiment.items():
        print(f"{sentiment}: {list(words.keys())[:10]}")
    
    # Visualization: Word frequency distribution
    plt.figure(figsize=(14, 8))
    
    words = list(top_customer_words.keys())[:20]
    counts = [top_customer_words[word] for word in words]
    
    plt.bar(words, counts)
    plt.title('Top 20 Customer Words Overall')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_customer_words.png")
    
    # Word clouds for each sentiment
    plt.figure(figsize=(15, 5))
    
    sentiments = all_data['customer_sentiment'].unique()
    for i, sentiment in enumerate(sentiments):
        plt.subplot(1, len(sentiments), i+1)
        
        # Get words for this sentiment
        words = customer_words_by_sentiment[sentiment]
        
        if words:
            # Create and generate a word cloud image
            wordcloud = WordCloud(width=400, height=400, 
                                 background_color='white',
                                 max_words=50).generate_from_frequencies(words)
            
            # Display the word cloud
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f'{sentiment.capitalize()} Sentiment')
        else:
            plt.text(0.5, 0.5, f"No words for {sentiment}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_wordclouds.png")
    
    # TF-IDF analysis to find distinctive words
    print("\nComputing TF-IDF to find distinctive words by sentiment...")
    sentiment_differential_words = compute_tf_idf_by_sentiment(
        all_data, 'preprocessed_customer_text', 'customer_sentiment'
    )
    
    # Print top distinctive words by sentiment
    print("\nTop distinctive words by sentiment (from TF-IDF):")
    for sentiment, words in sentiment_differential_words.items():
        print(f"\n{sentiment}: {list(words.keys())[:10]}")
    
    # Save TF-IDF results
    sentiment_top_words = {k: list(v.keys())[:15] for k, v in sentiment_differential_words.items()}
    save_to_json(sentiment_top_words, "tfidf_distinctive_words.json")
    
    # Visualize TF-IDF distinctive words
    plt.figure(figsize=(15, 10))
    plt.suptitle('Distinctive Words by Sentiment (TF-IDF)', fontsize=16)
    
    sentiments = list(sentiment_differential_words.keys())
    num_sentiments = len(sentiments)
    
    for i, sentiment in enumerate(sentiments):
        if sentiment_differential_words[sentiment]:
            plt.subplot(1, num_sentiments, i+1)
            
            words = list(sentiment_differential_words[sentiment].keys())[:10]
            scores = list(sentiment_differential_words[sentiment].values())[:10]
            
            plt.barh(words, scores)
            plt.title(f'{sentiment.capitalize()} Sentiment')
            plt.xlabel('Distinctiveness Score')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/tfidf_distinctive_words.png")
    
    # Improved N-gram analysis using refined get_ngrams function
    print("\nPerforming N-gram analysis...")
    
    # Extract bigrams and trigrams from all customer text using improved method
    # We'll collect all n-grams first for each sentiment
    bigram_by_sentiment = {}
    trigram_by_sentiment = {}
    
    for sentiment in all_data['customer_sentiment'].unique():
        # Get all texts for this sentiment
        sentiment_texts = all_data[all_data['customer_sentiment'] == sentiment]['customer_text']
        
        # Extract bigrams and trigrams
        all_bigrams = []
        all_trigrams = []
        
        for text in sentiment_texts:
            # Get cleaned text using our enhanced method
            clean_text = ' '.join(preprocess_for_word_analysis(text))
            
            # Extract n-grams
            all_bigrams.extend(get_ngrams(clean_text, n=2))
            all_trigrams.extend(get_ngrams(clean_text, n=3))
        
        # Get most common n-grams
        bigram_by_sentiment[sentiment] = dict(Counter(all_bigrams).most_common(15))
        trigram_by_sentiment[sentiment] = dict(Counter(all_trigrams).most_common(15))
    
    print("\nTop bigrams by sentiment:")
    for sentiment, bigrams in bigram_by_sentiment.items():
        print(f"\n{sentiment}: {list(bigrams.keys())[:5]}")
    
    # Save N-gram results
    ngram_results = {
        "bigrams": {k: list(v.keys())[:10] for k, v in bigram_by_sentiment.items()},
        "trigrams": {k: list(v.keys())[:10] for k, v in trigram_by_sentiment.items()}
    }
    save_to_json(ngram_results, "ngram_analysis.json")
    
    # Also save the newer format to n_grams_by_sentiment.json for consistency
    with open(f"{output_dir}/n_grams_by_sentiment.json", 'w') as f:
        json.dump(ngram_results, f, indent=4)
    
    # PART 5: Sentiment Change Analysis
    print("\nAnalyzing sentiment changes within conversations...")
    
    # Process sentiment changes
    all_data['sentiment_analysis'] = all_data['conversation'].apply(analyze_sentiment_changes)
    
    # Extract sentiment metrics
    all_data['customer_initial_sentiment'] = all_data['sentiment_analysis'].apply(lambda x: x['customer_initial_sentiment'])
    all_data['customer_final_sentiment'] = all_data['sentiment_analysis'].apply(lambda x: x['customer_final_sentiment'])
    all_data['agent_initial_sentiment'] = all_data['sentiment_analysis'].apply(lambda x: x['agent_initial_sentiment'])
    all_data['agent_final_sentiment'] = all_data['sentiment_analysis'].apply(lambda x: x['agent_final_sentiment'])
    all_data['sentiment_volatility'] = all_data['sentiment_analysis'].apply(lambda x: x['sentiment_volatility'])
    all_data['avg_sentiment_change'] = all_data['sentiment_analysis'].apply(lambda x: x['avg_sentiment_change'])
    all_data['sentiment_trend'] = all_data['sentiment_analysis'].apply(lambda x: x['sentiment_trend'])
    
    # Calculate sentiment shift (final - initial)
    all_data['customer_sentiment_shift'] = all_data['customer_final_sentiment'] - all_data['customer_initial_sentiment']
    
    # Analyze sentiment metrics by final sentiment category
    sentiment_metrics_by_category = all_data.groupby('sentiment_cat')[
        ['customer_initial_sentiment', 'customer_final_sentiment', 
         'agent_initial_sentiment', 'agent_final_sentiment',
         'sentiment_volatility', 'avg_sentiment_change', 'customer_sentiment_shift']
    ].mean().reset_index()
    
    print("\nSentiment change metrics by final sentiment category:")
    print(sentiment_metrics_by_category)
    save_to_csv(sentiment_metrics_by_category, "sentiment_change_metrics.csv")
    
    # Analyze sentiment trend distribution
    sentiment_trend_dist = all_data.groupby(['sentiment_cat', 'sentiment_trend']).size().reset_index(name='count')
    print("\nSentiment trend distribution by category:")
    print(sentiment_trend_dist)
    save_to_csv(sentiment_trend_dist, "sentiment_trend_distribution.csv")
    
    # Visualize sentiment metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.barplot(x='sentiment_cat', y='customer_sentiment_shift', data=sentiment_metrics_by_category, palette='viridis')
    plt.title('Average Sentiment Shift by Category')
    plt.ylabel('Final - Initial Sentiment')
    
    plt.subplot(1, 3, 2)
    sns.barplot(x='sentiment_cat', y='sentiment_volatility', data=sentiment_metrics_by_category, palette='viridis')
    plt.title('Sentiment Volatility by Category')
    plt.ylabel('Standard Deviation of Sentiment')
    
    plt.subplot(1, 3, 3)
    # Create a melted dataframe for comparison
    sentiment_scores = sentiment_metrics_by_category[
        ['sentiment_cat', 'customer_initial_sentiment', 'customer_final_sentiment']
    ]
    sentiment_scores_melted = pd.melt(
        sentiment_scores, 
        id_vars=['sentiment_cat'],
        value_vars=['customer_initial_sentiment', 'customer_final_sentiment'],
        var_name='Measurement',
        value_name='Sentiment Score'
    )
    sentiment_scores_melted['Measurement'] = sentiment_scores_melted['Measurement'].map({
        'customer_initial_sentiment': 'Initial',
        'customer_final_sentiment': 'Final'
    })
    
    sns.barplot(x='sentiment_cat', y='Sentiment Score', hue='Measurement', 
                data=sentiment_scores_melted, palette='viridis')
    plt.title('Initial vs. Final Sentiment')
    plt.ylabel('Average Sentiment Score')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_change_analysis.png")
    
    # Visualize sentiment trend distribution
    plt.figure(figsize=(10, 6))
    trend_pivot = pd.crosstab(sentiment_trend_dist['sentiment_cat'], sentiment_trend_dist['sentiment_trend'], 
                             values=sentiment_trend_dist['count'], aggfunc='sum', normalize='index') * 100
    trend_pivot.plot(kind='bar', stacked=True)
    plt.title('Sentiment Trend Distribution by Category')
    plt.ylabel('Percentage')
    plt.xlabel('Sentiment Category')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_trend_distribution.png")
    
    # PART 6: Correlation Analysis and Feature Importance
    print("\nPerforming correlation analysis and feature importance...")
    
    # Select numeric features for correlation analysis
    numeric_features = [
        'conversation_length', 'customer_text_length', 'agent_text_length',
        'customer_word_count', 'agent_word_count', 'customer_sentence_count',
        'agent_sentence_count', 'customer_question_marks', 'customer_exclamation_marks',
        'customer_turns', 'agent_turns', 'total_turns', 'turn_ratio',
        'customer_initial_sentiment', 'customer_final_sentiment',
        'sentiment_volatility', 'customer_sentiment_shift'
    ]
    
    # Compute correlation matrix
    correlation_matrix = all_data[numeric_features].corr()
    save_to_csv(correlation_matrix, "correlation_matrix.csv")
    
    # Visualize correlation matrix
    plt.figure(figsize=(18, 15))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                mask=mask, square=True, linewidths=.5)
    plt.title('Correlation Between Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    
    # Feature importance analysis with RandomForest
    features = all_data[numeric_features].fillna(0)
    
    # Train a RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        clf.fit(features, all_data['customer_sentiment'])
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature importance for sentiment prediction:")
        print(feature_importance)
        save_to_csv(feature_importance, "feature_importance.csv")
        
        # Visualize feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance for Sentiment Prediction')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
        
    # Save metadata about the analysis
    metadata = {
        "dataset_size": len(all_data),
        "sentiment_distribution": sentiment_dist.to_dict(),
        "avg_conversation_length": float(all_data['conversation_length'].mean()),
        "avg_customer_text_length": float(all_data['customer_text_length'].mean()),
        "avg_agent_text_length": float(all_data['agent_text_length'].mean()),
        "avg_customer_turns": float(all_data['customer_turns'].mean()),
        "avg_agent_turns": float(all_data['agent_turns'].mean()),
        "top_distinctive_words_by_sentiment": sentiment_top_words,
        "analysis_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_to_json(metadata, "analysis_metadata.json")
    
    # Create a summary of key findings
    key_findings = [
        f"Dataset contains {len(all_data)} conversations with sentiment distribution: {dict(sentiment_dist)}",
        f"Negative sentiment conversations are typically longer ({text_stats_by_sentiment[text_stats_by_sentiment['sentiment_cat'] == 'Negative']['conversation_length'].values[0]:.1f} chars vs. {text_stats_by_sentiment[text_stats_by_sentiment['sentiment_cat'] == 'Neutral']['conversation_length'].values[0]:.1f} chars for neutral)",
        f"Negative sentiment conversations have more customer turns ({flow_by_sentiment[flow_by_sentiment['sentiment_cat'] == 'Negative']['customer_turns'].values[0]:.1f} vs. {flow_by_sentiment[flow_by_sentiment['sentiment_cat'] == 'Neutral']['customer_turns'].values[0]:.1f} for neutral)",
        f"Most important features for sentiment prediction: {', '.join(feature_importance['Feature'].head(3).tolist())}",
        f"Top distinctive words for negative sentiment: {', '.join(sentiment_top_words.get('negative', [])[:5])}",
        f"Top distinctive words for positive sentiment: {', '.join(sentiment_top_words.get('positive', [])[:5])}"
    ]
    
    save_to_txt('\n'.join(key_findings), "key_findings.txt")
    
    print("\nComprehensive data analysis completed successfully!")
    print(f"All outputs saved to '{output_dir}' directory")

# Artık explore_llm_data() içindeki preprocess_for_word_analysis fonksiyonunu çıkaralım
# ve yerine global tanımlı olanı kullanalım

def explore_llm_data(train_df):
    """Analyze the dataset for LLM fine-tuning and generate findings"""
    print("\nExploring data for LLM fine-tuning...")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import re
    import json
    import numpy as np
    from collections import Counter
    
    # Create folder for LLM exploration results
    EXPLORATION_DIR = os.path.join('data/llm_ready', 'data_exploration')
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    
    # Examine columns
    print("\nDataset columns:")
    print(train_df.columns.tolist())
    
    # Data types
    print("\nData types:")
    print(train_df.dtypes)
    
    # Sentiment class distribution
    sentiment_dist = train_df['customer_sentiment'].value_counts()
    print("\nSentiment class distribution:")
    print(sentiment_dist)
    
    # Sentiment class distribution chart
    plt.figure(figsize=(10, 6))
    sns.countplot(x='customer_sentiment', data=train_df)
    plt.title('Sentiment Class Distribution')
    plt.savefig(os.path.join(EXPLORATION_DIR, 'sentiment_distribution.png'))
    
    # Feature engineering: Conversation features
    train_df['conversation_length'] = train_df['conversation'].apply(len)
    train_df['word_count'] = train_df['conversation'].apply(lambda x: len(x.split()))
    train_df['sentence_count'] = train_df['conversation'].apply(lambda x: len(re.split(r'[.!?]+', x)))
    train_df['customer_lines'] = train_df['conversation'].apply(lambda x: len([l for l in x.split('\n') if l.strip().startswith('Customer:')]))
    train_df['agent_lines'] = train_df['conversation'].apply(lambda x: len([l for l in x.split('\n') if l.strip().startswith('Agent:')]))
    train_df['avg_customer_line_length'] = train_df.apply(
        lambda x: np.mean([len(l) for l in x['conversation'].split('\n') if l.strip().startswith('Customer:')]) 
        if x['customer_lines'] > 0 else 0, axis=1
    )
    
    # Analyze conversation features by sentiment
    print("\nConversation features (by sentiment):")
    conversation_stats = train_df.groupby('customer_sentiment')[
        ['conversation_length', 'word_count', 'sentence_count', 
         'customer_lines', 'agent_lines', 'avg_customer_line_length']
    ].describe()
    print(conversation_stats)
    
    # Conversation length vs sentiment relationship chart
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='customer_sentiment', y='conversation_length', data=train_df)
    plt.title('Conversation Length and Sentiment Relationship')
    plt.savefig(os.path.join(EXPLORATION_DIR, 'conversation_length_sentiment.png'))
    
    # Categorical variables and sentiment relationship
    if 'issue_area' in train_df.columns:
        # Issue area and sentiment relationship
        issue_sentiment = pd.crosstab(train_df['issue_area'], train_df['customer_sentiment'], normalize='index') * 100
        
        # Create the chart
        plt.figure(figsize=(14, 8))
        sns.heatmap(issue_sentiment, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Issue Area and Sentiment Relationship (%)')
        plt.savefig(os.path.join(EXPLORATION_DIR, 'issue_sentiment_relationship.png'))
    
    # Correlation analysis
    numeric_cols = ['conversation_length', 'word_count', 'sentence_count', 
                    'customer_lines', 'agent_lines', 'avg_customer_line_length']
    corr_matrix = train_df[numeric_cols].corr()
    
    # Correlation matrix chart
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Features')
    plt.savefig(os.path.join(EXPLORATION_DIR, 'correlation_matrix.png'))
    
    # Extract customer text for better analysis
    def extract_customer_text_llm(conversation):
        """Extract only the customer parts from the conversation"""
        lines = conversation.split('\n')
        customer_lines = []
        is_customer = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Customer:'):
                is_customer = True
                customer_text = line[9:].strip()
                if customer_text:
                    customer_lines.append(customer_text)
            elif line.startswith('Agent:'):
                is_customer = False
            elif is_customer and line:
                customer_lines.append(line)
        
        return ' '.join(customer_lines)
    
    train_df['customer_text'] = train_df['conversation'].apply(extract_customer_text_llm)
    
    # Most frequent meaningful words for each sentiment class
    word_freq_by_sentiment = {}
    for sentiment in train_df['customer_sentiment'].unique():
        texts = train_df[train_df['customer_sentiment'] == sentiment]['customer_text']
        all_words = []
        for text in texts:
            all_words.extend(preprocess_for_word_analysis(text))
        word_freq_by_sentiment[sentiment] = Counter(all_words).most_common(30)  # Increase from 20 to 30
    
    # Save most frequent words
    with open(os.path.join(EXPLORATION_DIR, 'common_words_by_sentiment.json'), 'w') as f:
        # Convert to a more readable format
        json_compatible = {sentiment: {word: count for word, count in freq} 
                         for sentiment, freq in word_freq_by_sentiment.items()}
        json.dump(json_compatible, f, indent=2)
    
    # Add a visualization for most distinctive words
    plt.figure(figsize=(15, 5))
    for i, sentiment in enumerate(word_freq_by_sentiment.keys()):
        plt.subplot(1, len(word_freq_by_sentiment), i+1)
        
        words = [word for word, _ in word_freq_by_sentiment[sentiment][:15]]
        counts = [count for _, count in word_freq_by_sentiment[sentiment][:15]]
        
        plt.barh(words, counts)
        plt.title(f'Top Words: {sentiment.capitalize()}')
        plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, 'top_words_by_sentiment.png'))
    
    # Generate and save n-grams
    bigram_by_sentiment = {}
    trigram_by_sentiment = {}
    
    for sentiment in train_df['customer_sentiment'].unique():
        sentiment_texts = train_df[train_df['customer_sentiment'] == sentiment]['customer_text']
        
        # Extract bigrams and trigrams
        all_bigrams = []
        all_trigrams = []
        
        for text in sentiment_texts:
            # Extract n-grams directly from text, with improved function
            all_bigrams.extend(get_ngrams(text, n=2))
            all_trigrams.extend(get_ngrams(text, n=3))
        
        # Get most common n-grams
        bigram_by_sentiment[sentiment] = dict(Counter(all_bigrams).most_common(20))  # Increase from 15 to 20
        trigram_by_sentiment[sentiment] = dict(Counter(all_trigrams).most_common(20))
    
    # Save n-grams
    ngram_results = {
        "bigrams": {k: list(v.keys())[:15] for k, v in bigram_by_sentiment.items()},
        "trigrams": {k: list(v.keys())[:15] for k, v in trigram_by_sentiment.items()}
    }
    
    with open(os.path.join(EXPLORATION_DIR, 'n_grams_by_sentiment.json'), 'w') as f:
        json.dump(ngram_results, f, indent=2)
    
    # Add a visualization for top n-grams
    plt.figure(figsize=(15, 10))
    
    # Bigrams visualization
    plt.subplot(2, 1, 1)
    plt.suptitle('Top Bigrams by Sentiment', fontsize=16)
    
    # Create a combined plot for bigrams
    bigram_data = []
    sentiments = []
    values = []
    
    for sentiment, bigrams in bigram_by_sentiment.items():
        top_bigrams = list(bigrams.items())[:8]  # Top 8 for readability
        for bigram, count in top_bigrams:
            bigram_data.append(bigram)
            sentiments.append(sentiment)
            values.append(count)
    
    # Convert to DataFrame for easier plotting
    bigram_df = pd.DataFrame({
        'bigram': bigram_data,
        'sentiment': sentiments,
        'count': values
    })
    
    # Plot bigrams
    g = sns.barplot(x='count', y='bigram', hue='sentiment', data=bigram_df)
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.legend(title='Sentiment')
    
    # Trigrams visualization
    plt.subplot(2, 1, 2)
    plt.suptitle('Top Trigrams by Sentiment', fontsize=16, y=0.98)
    
    # Create a combined plot for trigrams
    trigram_data = []
    sentiments = []
    values = []
    
    for sentiment, trigrams in trigram_by_sentiment.items():
        top_trigrams = list(trigrams.items())[:5]  # Top 5 for readability
        for trigram, count in top_trigrams:
            trigram_data.append(trigram)
            sentiments.append(sentiment)
            values.append(count)
    
    # Convert to DataFrame for easier plotting
    trigram_df = pd.DataFrame({
        'trigram': trigram_data,
        'sentiment': sentiments,
        'count': values
    })
    
    # Plot trigrams
    g = sns.barplot(x='count', y='trigram', hue='sentiment', data=trigram_df)
    plt.xlabel('Frequency')
    plt.ylabel('Trigram')
    plt.legend(title='Sentiment')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, 'top_ngrams_visual.png'))
    
    # Customer-agent dynamics analysis
    train_df['customer_agent_ratio'] = train_df['customer_lines'] / train_df['agent_lines'].replace(0, 1)
    
    # Dynamics vs sentiment relationship chart
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='customer_sentiment', y='customer_agent_ratio', data=train_df)
    plt.title('Customer-Agent Conversation Ratio and Sentiment Relationship')
    plt.savefig(os.path.join(EXPLORATION_DIR, 'customer_agent_ratio_sentiment.png'))
    
    # Summarize findings
    findings = {
        "sentiment_distribution": sentiment_dist.to_dict(),
        "conversation_features": {
            "by_sentiment": {sentiment: {
                "avg_length": train_df[train_df['customer_sentiment']==sentiment]['conversation_length'].mean(),
                "avg_words": train_df[train_df['customer_sentiment']==sentiment]['word_count'].mean(),
                "avg_sentences": train_df[train_df['customer_sentiment']==sentiment]['sentence_count'].mean(),
                "avg_customer_lines": train_df[train_df['customer_sentiment']==sentiment]['customer_lines'].mean(),
                "avg_agent_lines": train_df[train_df['customer_sentiment']==sentiment]['agent_lines'].mean(),
            } for sentiment in train_df['customer_sentiment'].unique()}
        },
        "correlations": {
            f"{col1}_{col2}": corr_matrix.loc[col1, col2]
            for col1 in numeric_cols for col2 in numeric_cols if col1 != col2
        },
        "key_findings": [
            "There is an imbalanced distribution among sentiment classes.",
            "Conversations with negative sentiment tend to be longer.",
            "There is a relationship between customer speech length and sentiment.",
            "Customer-agent interaction dynamics differ across sentiment classes."
        ],
        "distinctive_words": {
            sentiment: [word for word, _ in word_freq_by_sentiment.get(sentiment, [])[:10]]
            for sentiment in train_df['customer_sentiment'].unique()
        },
        "distinctive_ngrams": {
            sentiment: {
                "bigrams": list(bigram_by_sentiment.get(sentiment, {}).keys())[:5],
                "trigrams": list(trigram_by_sentiment.get(sentiment, {}).keys())[:5]
            }
            for sentiment in train_df['customer_sentiment'].unique()
        }
    }
    
    # Save findings as JSON
    with open(os.path.join(EXPLORATION_DIR, 'findings.json'), 'w') as f:
        json.dump(findings, f, indent=2)
    
    print("\nLLM data exploration completed. Results saved.")
    
    return train_df, findings

# Bridge function - preprocess_for_llm.py will call this function
def get_exploration_findings(train_df):
    """Get exploration findings for the LLM data preprocessing pipeline.
    This function is called from preprocess_for_llm.py"""
    
    # First, run the full exploration (main exploration) - if not done before
    if not os.path.exists('data_preprocessing/data_exploration'):
        main()
    
    # Then specialized exploration for LLM - create output directory
    EXPLORATION_DIR = os.path.join('data/llm_ready', 'data_exploration')
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    
    train_df, findings = explore_llm_data(train_df)
    
    # Copy key results from main exploration folder to LLM exploration folder
    # This way, the LLM exploration folder will have the improved n-gram and word analyses
    try:
        import shutil
        for file in ['customer_words_by_sentiment.json', 'ngram_analysis.json', 'n_grams_by_sentiment.json']:
            source = os.path.join('data_preprocessing/data_exploration', file)
            dest = os.path.join(EXPLORATION_DIR, file)
            if os.path.exists(source):
                shutil.copy2(source, dest)
                print(f"Copied {file} to LLM exploration directory")
    except Exception as e:
        print(f"Warning: Could not copy files: {e}")
    
    return train_df, findings

if __name__ == "__main__":
    main() 