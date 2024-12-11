from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import sqlite3

def setup_chroma():
    # Initialize Chroma client with persistence
    client = Client(Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    ))
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="snippets",
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity
    )
    
    return collection

def embed_snippets(db_path):
    # Initialize SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get Chroma collection
    collection = setup_chroma()
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Add column for chroma_id if it doesn't exist
    cursor.execute('''
        ALTER TABLE snippets 
        ADD COLUMN chroma_id TEXT;
    ''')
    conn.commit()
    
    # Get all snippets that haven't been embedded yet
    cursor.execute('SELECT id, content FROM snippets WHERE chroma_id IS NULL')
    snippets = cursor.fetchall()
    
    # Process snippets in batches
    batch_size = 100
    for i in range(0, len(snippets), batch_size):
        batch = snippets[i:i + batch_size]
        
        # Prepare data for Chroma
        ids = [str(snippet[0]) for snippet in batch]
        texts = [snippet[1] for snippet in batch]
        embeddings = model.encode(texts).tolist()
        
        # Add to Chroma
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings
        )
        
        # Update SQLite with Chroma IDs
        for snippet_id in ids:
            cursor.execute(
                'UPDATE snippets SET chroma_id = ? WHERE id = ?',
                (snippet_id, snippet_id)
            )
        
        conn.commit()
    
    conn.close()

def search_similar_snippets(query, n_results=5):
    # Initialize SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get Chroma collection
    collection = setup_chroma()
    
    # Encode query
    query_embedding = model.encode(query).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results
