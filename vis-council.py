import sqlite3
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

def get_council_members() -> List[Tuple[str, str, str, str]]:
    """Get all council members from the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT location_name, location_state, member_name, position 
        FROM council_members
    ''')
    
    members = cursor.fetchall()
    conn.close()
    return members

def get_meeting_transcripts() -> List[Tuple[str, str, str, str, str]]:
    """Get all meeting transcripts"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT meeting_id, transcript, location_name, location_state, date 
        FROM meetings 
        WHERE transcript IS NOT NULL
    ''')
    
    transcripts = cursor.fetchall()
    conn.close()
    return transcripts

def create_name_variants(full_name: str) -> List[str]:
    """Create variations of a name (first, last, full)"""
    names = full_name.strip().split()
    variants = []
    
    if len(names) >= 2:
        # Add full name
        variants.append(full_name.lower())
        # Add last name
        variants.append(names[-1].lower())
        # Add first name
        variants.append(names[0].lower())
        # Add first + last (if there's a middle name)
        if len(names) > 2:
            variants.append(f"{names[0]} {names[-1]}".lower())
    
    return variants

def count_mentions(transcript: str, name_variants: List[str]) -> int:
    """Count mentions of name variants in transcript"""
    transcript = transcript.lower()
    total_mentions = 0
    
    for variant in name_variants:
        # Look for the name with word boundaries
        pattern = r'\b' + re.escape(variant) + r'\b'
        mentions = len(re.findall(pattern, transcript))
        total_mentions += mentions
    
    return total_mentions

def store_mention_results(results: Dict):
    """Store the mention results in the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS council_member_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER,
            location_name TEXT,
            location_state TEXT,
            member_name TEXT,
            mention_count INTEGER,
            meeting_date TEXT,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        )
    ''')
    
    # Store results
    for (meeting_id, location, state, date), member_counts in results.items():
        for member_name, count in member_counts.items():
            cursor.execute('''
                INSERT INTO council_member_mentions 
                (meeting_id, location_name, location_state, member_name, mention_count, meeting_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (meeting_id, location, state, member_name, count, date))
    
    conn.commit()
    conn.close()

def main():
    # Get all council members and their locations
    print("Fetching council members...")
    council_members = get_council_members()
    
    # Get all meeting transcripts
    print("Fetching meeting transcripts...")
    transcripts = get_meeting_transcripts()
    
    # Create a dictionary to store results
    results = defaultdict(lambda: defaultdict(int))
    aggregated_results = defaultdict(lambda: defaultdict(int))
    
    # Process each transcript
    print("Analyzing mentions...")
    for meeting_id, transcript, location, state, date in tqdm(transcripts, desc="Processing transcripts"):
        # Get relevant council members for this location
        relevant_members = [m for m in council_members 
                          if m[0] == location and m[1] == state]
        
        for member in relevant_members:
            member_name = member[2]  # Full name
            name_variants = create_name_variants(member_name)
            
            # Count mentions in this transcript
            mentions = count_mentions(transcript, name_variants)
            
            if mentions > 0:
                # Store per-meeting results
                results[(meeting_id, location, state, date)][member_name] = mentions
                # Store aggregated results
                aggregated_results[(location, state)][member_name] += mentions
    
    # Store results in database
    print("Storing results...")
    store_mention_results(results)
    
    # Print summary
    print("\nMention Summary:")
    for (location, state), member_counts in aggregated_results.items():
        print(f"\n{location}, {state}:")
        for member_name, count in sorted(member_counts.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {member_name}: {count} mentions")

if __name__ == "__main__":
    main()
