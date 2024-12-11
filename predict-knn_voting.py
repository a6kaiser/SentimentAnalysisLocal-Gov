import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

class VotePredictorKNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.topics_collection = self.client.get_collection("meeting_topics")
    
    def get_historical_votes(self) -> List[Tuple[int, str, bool]]:
        """Get all historical votes with their outcomes"""
        conn = sqlite3.connect('meetings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.id, t.name, v.did_pass
            FROM votes v
            JOIN vote_topic_matches m ON v.id = m.vote_id
            JOIN topics t ON m.topic_id = t.id
            WHERE v.did_pass IS NOT NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def predict(self, prompts: List[str]) -> List[Tuple[bool, float]]:
        """
        Predict vote outcomes for a list of prompts.
        Returns list of (prediction, confidence) tuples.
        """
        # Get historical data
        historical_votes = self.get_historical_votes()
        if not historical_votes:
            raise ValueError("No historical vote data available")
        
        # Embed the prompts
        prompt_embeddings = self.model.encode(prompts)
        
        # Get historical topic embeddings
        historical_topics = [vote[1] for vote in historical_votes]
        historical_embeddings = self.model.encode(historical_topics)
        
        predictions = []
        for prompt_embedding in tqdm(prompt_embeddings, desc="Predicting outcomes"):
            # Calculate cosine similarities
            similarities = np.dot(historical_embeddings, prompt_embedding) / (
                np.linalg.norm(historical_embeddings, axis=1) * np.linalg.norm(prompt_embedding)
            )
            
            # Get k nearest neighbors
            k_nearest_indices = np.argsort(similarities)[-self.k:]
            k_nearest_outcomes = [historical_votes[i][2] for i in k_nearest_indices]
            k_nearest_similarities = similarities[k_nearest_indices]
            
            # Weight votes by similarity
            weighted_votes = np.sum(k_nearest_outcomes * k_nearest_similarities)
            weighted_total = np.sum(k_nearest_similarities)
            
            # Calculate weighted probability of passing
            pass_probability = weighted_votes / weighted_total
            
            # Make prediction with confidence
            will_pass = pass_probability >= 0.5
            confidence = abs(pass_probability - 0.5) * 2  # Scale to 0-1
            
            predictions.append((will_pass, confidence))
        
        return predictions
    
    def get_similar_historical_votes(self, prompt: str, n: int = 5) -> List[dict]:
        """Get similar historical votes for explanation"""
        conn = sqlite3.connect('meetings.db')
        cursor = conn.cursor()
        
        # Get historical votes with more details
        cursor.execute('''
            SELECT 
                v.id,
                t.name,
                v.did_pass,
                v.votes_for,
                v.votes_against,
                v.votes_abstain,
                v.meeting_id
            FROM votes v
            JOIN vote_topic_matches m ON v.id = m.vote_id
            JOIN topics t ON m.topic_id = t.id
            WHERE v.did_pass IS NOT NULL
        ''')
        
        historical_votes = cursor.fetchall()
        conn.close()
        
        if not historical_votes:
            return []
        
        # Embed prompt and historical topics
        prompt_embedding = self.model.encode([prompt])[0]
        historical_topics = [vote[1] for vote in historical_votes]
        historical_embeddings = self.model.encode(historical_topics)
        
        # Calculate similarities
        similarities = np.dot(historical_embeddings, prompt_embedding) / (
            np.linalg.norm(historical_embeddings, axis=1) * np.linalg.norm(prompt_embedding)
        )
        
        # Get top N similar votes
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        similar_votes = []
        for idx in top_indices:
            vote = historical_votes[idx]
            similar_votes.append({
                'topic': vote[1],
                'did_pass': vote[2],
                'votes_for': vote[3],
                'votes_against': vote[4],
                'votes_abstain': vote[5],
                'meeting_id': vote[6],
                'similarity': similarities[idx]
            })
        
        return similar_votes

def main():
    # Example usage
    predictor = VotePredictorKNN(k=5)
    
    # Example prompts
    test_prompts = [
        "Approval of new zoning regulations for downtown development",
        "Budget allocation for road maintenance",
        "Proposal to increase parking fees"
    ]
    
    # Make predictions
    predictions = predictor.predict(test_prompts)
    
    # Print predictions with explanations
    for prompt, (will_pass, confidence) in zip(test_prompts, predictions):
        print(f"\nPrompt: {prompt}")
        print(f"Prediction: {'PASS' if will_pass else 'FAIL'}")
        print(f"Confidence: {confidence:.2f}")
        
        print("\nSimilar historical votes:")
        similar_votes = predictor.get_similar_historical_votes(prompt)
        for i, vote in enumerate(similar_votes, 1):
            print(f"\n{i}. Topic: {vote['topic']}")
            print(f"   Outcome: {'PASSED' if vote['did_pass'] else 'FAILED'}")
            print(f"   Votes: {vote['votes_for']} for, {vote['votes_against']} against, "
                  f"{vote['votes_abstain']} abstain")
            print(f"   Similarity: {vote['similarity']:.2f}")
            print(f"   Meeting ID: {vote['meeting_id']}")

if __name__ == "__main__":
    main()
