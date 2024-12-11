import sqlite3
import chromadb
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json

def init_match_table():
    """Initialize the table for storing vote-topic matches"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vote_topic_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            vote_id INTEGER,
            topic_id INTEGER,
            confidence_score FLOAT,
            is_flagged BOOLEAN,
            flag_reason TEXT,
            FOREIGN KEY(vote_id) REFERENCES votes(id),
            FOREIGN KEY(topic_id) REFERENCES topics(id),
            UNIQUE(vote_id, topic_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_meeting_votes_and_topics():
    """Get all votes and topics grouped by meeting_id"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT meeting_id, 
               GROUP_CONCAT(id || ',' || name) as votes
        FROM votes
        GROUP BY meeting_id
    ''')
    votes_by_meeting = {row[0]: [(int(v.split(',')[0]), v.split(',')[1]) 
                                for v in row[1].split('||')] 
                       for row in cursor.fetchall()}
    
    cursor.execute('''
        SELECT meeting_id, 
               GROUP_CONCAT(id || ',' || name) as topics
        FROM topics
        GROUP BY meeting_id
    ''')
    topics_by_meeting = {row[0]: [(int(t.split(',')[0]), t.split(',')[1]) 
                                 for t in row[1].split('||')] 
                        for row in cursor.fetchall()}
    
    conn.close()
    return votes_by_meeting, topics_by_meeting

def calculate_word_frequency_similarity(text1, text2):
    """Calculate similarity based on word frequency"""
    # Tokenize and get word frequencies
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())
    
    # Get common words
    common_words = set(words1.keys()) & set(words2.keys())
    
    if not common_words:
        return 0.0
    
    # Calculate similarity based on common word frequencies
    similarity = sum(min(words1[word], words2[word]) for word in common_words) / \
                max(sum(words1.values()), sum(words2.values()))
    
    return similarity

def main():
    # Initialize match table
    init_match_table()
    
    # Get Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")
    topics_collection = client.get_collection("meeting_topics")
    votes_collection = client.get_collection("meeting_votes")
    
    # Get votes and topics by meeting
    votes_by_meeting, topics_by_meeting = get_meeting_votes_and_topics()
    
    # Process each meeting
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    for meeting_id in tqdm(votes_by_meeting.keys()):
        if meeting_id not in topics_by_meeting:
            continue
        
        votes = votes_by_meeting[meeting_id]
        topics = topics_by_meeting[meeting_id]
        
        for vote_id, vote_name in votes:
            # Get embeddings for vote
            vote_results = votes_collection.query(
                query_texts=[vote_name],
                n_results=len(topics)
            )
            
            # Get topic embeddings for this meeting
            topic_names = [topic[1] for topic in topics]
            topic_results = topics_collection.query(
                query_texts=[vote_name],
                n_results=len(topics)
            )
            
            # Calculate word frequency similarities
            word_freq_scores = [
                calculate_word_frequency_similarity(vote_name, topic_name)
                for topic_name in topic_names
            ]
            
            # Get best matches from both methods
            embedding_best_idx = np.argmax(topic_results['distances'][0])
            word_freq_best_idx = np.argmax(word_freq_scores)
            
            # Get the matched topic
            matched_topic_id = topics[embedding_best_idx][0]
            confidence_score = float(topic_results['distances'][0][embedding_best_idx])
            
            # Flag if methods disagree
            is_flagged = embedding_best_idx != word_freq_best_idx
            flag_reason = None
            if is_flagged:
                flag_reason = (
                    f"Embedding match: '{topic_names[embedding_best_idx]}' "
                    f"Word freq match: '{topic_names[word_freq_best_idx]}'"
                )
            
            # Store the match
            try:
                cursor.execute('''
                    INSERT INTO vote_topic_matches 
                    (meeting_id, vote_id, topic_id, confidence_score, 
                     is_flagged, flag_reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    meeting_id,
                    vote_id,
                    matched_topic_id,
                    confidence_score,
                    is_flagged,
                    flag_reason
                ))
            except sqlite3.IntegrityError:
                # Update if already exists
                cursor.execute('''
                    UPDATE vote_topic_matches 
                    SET confidence_score = ?,
                        is_flagged = ?,
                        flag_reason = ?
                    WHERE vote_id = ? AND topic_id = ?
                ''', (
                    confidence_score,
                    is_flagged,
                    flag_reason,
                    vote_id,
                    matched_topic_id
                ))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
