import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import openai
import json

class CouncilSentimentAnalyzer:
    def __init__(self, k: int = 3):
        self.k = k
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.topics_collection = self.client.get_collection("meeting_topics")
        
        self.system_prompt = """You are an expert at analyzing local government meeting transcripts.
Your task is to analyze the sentiment and attitudes of council members regarding a specific topic.
Focus on:
1. Overall council sentiment (positive, negative, neutral, mixed)
2. Key concerns or support points raised
3. Notable disagreements or consensus
4. Changes in sentiment during discussion
5. Specific council member positions if mentioned"""

    def get_meeting_transcripts(self, meeting_ids: List[str]) -> List[Dict]:
        """Get transcripts for specified meeting IDs"""
        conn = sqlite3.connect('meetings.db')
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(meeting_ids))
        cursor.execute(f'''
            SELECT 
                m.meeting_id,
                m.transcript,
                m.location_name,
                m.date,
                m.title
            FROM meetings m
            WHERE m.meeting_id IN ({placeholders})
            AND m.transcript IS NOT NULL
        ''', meeting_ids)
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'meeting_id': r[0],
            'transcript': r[1],
            'location': r[2],
            'date': r[3],
            'title': r[4]
        } for r in results]

    def find_relevant_meetings(self, prompt: str, location: str = None) -> List[str]:
        """Find relevant meeting IDs based on topic similarity"""
        # Query Chroma for similar topics
        results = self.topics_collection.query(
            query_texts=[prompt],
            n_results=self.k * 2  # Get more results to filter
        )
        
        # Get meeting IDs from metadata
        meeting_ids = [
            meta['meeting_id'] 
            for meta in results['metadatas'][0]
        ]
        
        # If location specified, filter results
        if location:
            conn = sqlite3.connect('meetings.db')
            cursor = conn.cursor()
            
            filtered_ids = []
            for mid in meeting_ids:
                cursor.execute('''
                    SELECT 1 FROM meetings 
                    WHERE meeting_id = ? 
                    AND location_name LIKE ?
                ''', (mid, f"%{location}%"))
                
                if cursor.fetchone():
                    filtered_ids.append(mid)
            
            conn.close()
            meeting_ids = filtered_ids[:self.k]
        else:
            meeting_ids = meeting_ids[:self.k]
        
        return meeting_ids

    def get_distinct_locations(self) -> List[str]:
        """Get all distinct locations from the database"""
        conn = sqlite3.connect('meetings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT location_name 
            FROM meetings 
            WHERE location_name IS NOT NULL
        ''')
        
        locations = [row[0] for row in cursor.fetchall()]
        conn.close()
        return locations

    def analyze_sentiment(self, prompt: str, location: str = None) -> Dict:
        """
        Analyze council sentiment towards a topic using relevant meeting transcripts.
        If location is None, analyze for all distinct locations.
        
        Returns:
            For single location:
            {
                'sentiment_analysis': str,  # GPT analysis of council sentiment
                'meetings': List[Dict],     # Relevant meetings found
                'prompt': str,              # Original prompt
                'location': str             # Location analyzed
            }
            
            For all locations:
            {
                'analyses_by_location': Dict[str, Dict],  # Location -> Analysis mapping
                'prompt': str,                            # Original prompt
                'locations_analyzed': List[str]           # All locations analyzed
            }
        """
        if location is not None:
            return self._analyze_single_location(prompt, location)
        
        # Analyze all locations
        locations = self.get_distinct_locations()
        print(f"Found {len(locations)} distinct locations to analyze")
        
        # Track successful and failed analyses
        all_analyses = {}
        failed_locations = []
        
        # Analyze each location
        for loc in tqdm(locations, desc="Analyzing locations"):
            analysis = self._analyze_single_location(prompt, loc)
            if 'error' in analysis:
                failed_locations.append((loc, analysis['error']))
                continue
            all_analyses[loc] = analysis
        
        # Handle case where no analyses were successful
        if not all_analyses:
            error_details = "\n".join([f"{loc}: {err}" for loc, err in failed_locations])
            return {
                'error': "Failed to analyze any locations",
                'error_details': error_details,
                'sentiment': None,
                'meetings': [],
                'summary': None
            }
        
        # Summarize results
        successful_count = len(all_analyses)
        failed_count = len(failed_locations)
        
        return {
            'analyses_by_location': all_analyses,
            'prompt': prompt,
            'locations_analyzed': list(all_analyses.keys()),
            'analysis_summary': {
                'total_locations': len(locations),
                'successful_analyses': successful_count,
                'failed_analyses': failed_count,
                'failed_locations': failed_locations if failed_locations else None
            }
        }

    def _analyze_single_location(self, prompt: str, location: str) -> Dict:
        """Internal method to analyze sentiment for a single location"""
        # Find relevant meetings
        meeting_ids = self.find_relevant_meetings(prompt, location)
        if not meeting_ids:
            return {
                'error': f"No relevant meetings found for topic in {location}",
                'sentiment': None,
                'meetings': [],
                'summary': None
            }
        
        # Get meeting transcripts
        meetings = self.get_meeting_transcripts(meeting_ids)
        
        # Prepare context for GPT
        context = f"Topic: {prompt}\nLocation: {location}\n\n"
        for meeting in meetings:
            context += f"Meeting Date: {meeting['date']}\n"
            context += f"Transcript Excerpt:\n{meeting['transcript'][:2000]}...\n\n"
        
        # Generate analysis using GPT
        analysis_prompt = f"""Analyze the council's sentiment and attitudes regarding this topic from the provided meeting transcripts.
Please provide:
1. Overall sentiment summary
2. Key points of support or concern
3. Notable council member positions
4. Any observed patterns or changes in sentiment

Context:
{context}"""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            
            return {
                'sentiment_analysis': analysis,
                'meetings': [{
                    'meeting_id': m['meeting_id'],
                    'date': m['date'],
                    'location': m['location'],
                    'title': m['title']
                } for m in meetings],
                'prompt': prompt,
                'location': location
            }
            
        except Exception as e:
            return {
                'error': f"Error analyzing sentiment: {str(e)}",
                'meetings': meetings,
                'prompt': prompt,
                'location': location
            }

def main():
    analyzer = CouncilSentimentAnalyzer(k=3)
    
    # Example usage
    test_prompts = [
        {
            'prompt': "Development of bike lanes",
            'location': "Akron"  # Specific location
        },
        {
            'prompt': "multifamily development",
            'location': None  # Will analyze all locations
        },
        {
            'prompt': "affordable housing",
            'location': None  # Will analyze all locations
        },
        {
            'prompt': "short term rentals",
            'location': None  # Will analyze all locations
        }
    ]
    
    for test in test_prompts:
        print(f"\nAnalyzing: {test['prompt']}")
        print(f"Location: {test['location'] or 'All locations'}")
        print("-" * 50)
        
        analysis = analyzer.analyze_sentiment(test['prompt'], test['location'])
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            continue
        
        if test['location'] is None:
            # Print analysis for all locations
            print(f"\nAnalyzed {len(analysis['locations_analyzed'])} locations:")
            for location, loc_analysis in analysis['analyses_by_location'].items():
                print(f"\n=== {location} ===")
                print("\nSentiment Analysis:")
                print(loc_analysis['sentiment_analysis'])
                
                print("\nRelevant Meetings:")
                for meeting in loc_analysis['meetings']:
                    print(f"\nDate: {meeting['date']}")
                    print(f"Title: {meeting['title']}")
                    print(f"Meeting ID: {meeting['meeting_id']}")
        else:
            # Print analysis for single location
            print("\nSentiment Analysis:")
            print(analysis['sentiment_analysis'])
            
            print("\nRelevant Meetings:")
            for meeting in analysis['meetings']:
                print(f"\nDate: {meeting['date']}")
                print(f"Location: {meeting['location']}")
                print(f"Title: {meeting['title']}")
                print(f"Meeting ID: {meeting['meeting_id']}")

if __name__ == "__main__":
    main()
