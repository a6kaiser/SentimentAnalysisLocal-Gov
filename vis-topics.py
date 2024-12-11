import sqlite3
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

def load_topics_data():
    """Load topics data from the database into a pandas DataFrame"""
    conn = sqlite3.connect('meetings.db')
    
    query = '''
        SELECT * FROM topics
    '''
    
    df = pd.read_sql_query(query, conn)
    
    # Parse JSON strings into lists
    df['speakers'] = df['speakers'].apply(json.loads)
    df['indicators'] = df['indicators'].apply(json.loads)
    
    conn.close()
    return df

def create_topic_wordcloud(df):
    """Generate a wordcloud from all topic names"""
    text = ' '.join(df['name'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topic Word Cloud')
    plt.tight_layout(pad=0)
    plt.savefig('topic_wordcloud.png')
    plt.close()

def main():
    # Load data
    print("Loading topics data...")
    df = load_topics_data()
    
    # Display sample of the data
    print("\nSample of 5 random topics:")
    print("=" * 80)
    sample = df.sample(n=5)
    for _, row in sample.iterrows():
        print(f"Topic: {row['name']}")
        print(f"Meeting ID: {row['meeting_id']}")
        print(f"Speakers: {', '.join(row['speakers'])}")
        print(f"Indicators: {', '.join(row['indicators'])}")
        print("-" * 80)
    
    # Generate wordcloud
    print("\nGenerating word cloud...")
    create_topic_wordcloud(df)
    print("Word cloud saved as 'topic_wordcloud.png'")

if __name__ == "__main__":
    main()
