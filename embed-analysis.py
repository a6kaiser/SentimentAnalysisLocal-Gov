import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

def get_topics_from_db():
    """Fetch all topics from the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, meeting_id, name 
        FROM topics
        WHERE name IS NOT NULL
    ''')
    
    topics = cursor.fetchall()
    conn.close()
    return topics

def get_votes_from_db():
    """Fetch all votes from the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, meeting_id, name 
        FROM votes
        WHERE name IS NOT NULL
    ''')
    
    votes = cursor.fetchall()
    conn.close()
    return votes

def main():
    # Initialize SBERT model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collections
    topics_collection = client.get_or_create_collection(
        name="meeting_topics",
        metadata={"description": "Meeting topic embeddings"}
    )
    
    votes_collection = client.get_or_create_collection(
        name="meeting_votes",
        metadata={"description": "Meeting vote embeddings"}
    )
    
    # Process topics
    topics = get_topics_from_db()
    print(f"Found {len(topics)} topics to embed")
    
    # Prepare topics data for batch processing
    topic_ids = [str(topic[0]) for topic in topics]
    topic_documents = [topic[2] for topic in topics]
    topic_metadatas = [{"meeting_id": topic[1]} for topic in topics]
    
    # Create topic embeddings in batches
    batch_size = 100
    for i in tqdm(range(0, len(topics), batch_size), desc="Embedding topics"):
        batch_ids = topic_ids[i:i + batch_size]
        batch_documents = topic_documents[i:i + batch_size]
        batch_metadatas = topic_metadatas[i:i + batch_size]
        
        # Add to Chroma
        topics_collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    
    # Process votes
    votes = get_votes_from_db()
    print(f"Found {len(votes)} votes to embed")
    
    # Prepare votes data for batch processing
    vote_ids = [str(vote[0]) for vote in votes]
    vote_documents = [vote[2] for vote in votes]
    vote_metadatas = [{"meeting_id": vote[1]} for vote in votes]
    
    # Create vote embeddings in batches
    for i in tqdm(range(0, len(votes), batch_size), desc="Embedding votes"):
        batch_ids = vote_ids[i:i + batch_size]
        batch_documents = vote_documents[i:i + batch_size]
        batch_metadatas = vote_metadatas[i:i + batch_size]
        
        # Add to Chroma
        votes_collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    
    print(f"Successfully embedded {len(topics)} topics and {len(votes)} votes")

if __name__ == "__main__":
    main()
