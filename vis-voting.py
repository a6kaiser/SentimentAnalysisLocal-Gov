import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

def load_voting_data():
    """Load voting data from the database into DataFrames"""
    conn = sqlite3.connect('meetings.db')
    
    # Get main votes table with dates
    votes_query = '''
        SELECT v.*, m.date
        FROM votes v
        LEFT JOIN meetings m ON v.meeting_id = m.meeting_id
    '''
    votes_df = pd.read_sql_query(votes_query, conn)
    
    # Get voting details table
    details_query = '''
        SELECT * FROM voting_details
    '''
    details_df = pd.read_sql_query(details_query, conn)
    
    # Parse JSON strings for indicators
    votes_df['indicators'] = votes_df['indicators'].apply(json.loads)
    
    conn.close()
    return votes_df, details_df

def create_pass_fail_pie(votes_df):
    """Create a pie chart of passed vs failed votes"""
    results = votes_df['did_pass'].value_counts()
    
    plt.figure(figsize=(10, 8))
    plt.pie(results.values, labels=['Passed' if i else 'Failed' for i in results.index], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Vote Outcomes Distribution')
    plt.savefig('vote_outcomes_pie.png')
    plt.close()

def create_contested_wordcloud(contested_votes):
    """Generate a wordcloud from contested vote names"""
    text = ' '.join(contested_votes['name'])
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='RdYlBu').generate(text)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Contested Votes Word Cloud')
    plt.tight_layout(pad=0)
    plt.savefig('contested_votes_wordcloud.png')
    plt.close()

def main():
    # Load data
    print("Loading voting data...")
    votes_df, details_df = load_voting_data()
    
    # Display sample of votes
    print("\nSample of 5 random votes:")
    print("=" * 80)
    sample = votes_df.sample(n=5)
    for _, row in sample.iterrows():
        print(f"Date: {row['date']}")
        print(f"Vote: {row['name']}")
        print(f"Meeting ID: {row['meeting_id']}")
        print(f"Outcome: {'Passed' if row['did_pass'] else 'Failed'}")
        print(f"Vote Count: For={row['votes_for']}, Against={row['votes_against']}, Abstain={row['votes_abstain']}")
        print(f"Indicators: {', '.join(row['indicators'])}")
        print("-" * 80)
    
    # Display all contested votes
    contested_votes = votes_df[votes_df['votes_against'] > 0].sort_values('date', ascending=False)
    print(f"\nAll Contested Votes ({len(contested_votes)} total):")
    print("=" * 80)
    for _, row in contested_votes.iterrows():
        print(f"Date: {row['date']}")
        print(f"Vote: {row['name']}")
        print(f"Outcome: {'Passed' if row['did_pass'] else 'Failed'}")
        print(f"Vote Count: For={row['votes_for']}, Against={row['votes_against']}, Abstain={row['votes_abstain']}")
        print("-" * 80)
    
    # Create wordcloud for contested votes
    if not contested_votes.empty:
        print("\nGenerating word cloud for contested votes...")
        create_contested_wordcloud(contested_votes)
        print("Word cloud saved as 'contested_votes_wordcloud.png'")
    
    # Display some voting details
    print("\nSample of 5 random individual votes:")
    print("=" * 80)
    sample_details = details_df.sample(n=5)
    for _, row in sample_details.iterrows():
        print(f"Vote ID: {row['vote_id']}")
        print(f"Voter: {row['voter']}")
        print(f"Vote Cast: {row['vote']}")
        print("-" * 80)
    
    # Generate pie chart
    print("\nGenerating vote outcomes pie chart...")
    create_pass_fail_pie(votes_df)
    print("Pie chart saved as 'vote_outcomes_pie.png'")
    
    # Print some summary statistics
    print("\nVoting Summary Statistics:")
    print("=" * 80)
    print(f"Total number of votes: {len(votes_df)}")
    print(f"Total passed: {votes_df['did_pass'].sum()}")
    print(f"Total failed: {len(votes_df) - votes_df['did_pass'].sum()}")
    print(f"Number of contested votes: {len(contested_votes)}")
    print(f"Average 'For' votes: {votes_df['votes_for'].mean():.2f}")
    print(f"Average 'Against' votes: {votes_df['votes_against'].mean():.2f}")
    print(f"Average 'Abstain' votes: {votes_df['votes_abstain'].mean():.2f}")

if __name__ == "__main__":
    main()
