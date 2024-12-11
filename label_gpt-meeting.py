import openai
import json
import sqlite3
from typing import Dict, Any
from tqdm import tqdm
import argparse

system_prompt = """You are a specialized assistant for local government meeting documentation. Your role is to:
1. Extract and structure key meeting information following official record-keeping standards
2. Identify all formal actions, discussions, and decisions
3. Maintain accurate attribution of speakers and votes
4. Flag any procedural or regulatory compliance items"""

prompt = '''Parse the following transcript and provide a JSON object with these fields:
{
    "topics": [{
        "name": "",
        "speakers": [""],
        "indicators": [""]
    }],
    "motions": [{
        "name": "",
        "initiator": "",
        "seconder": "",
        "opponents": [],
        "indicators": [""]
    }],
    "votes": [{
        "name": "",
        "didPass": boolean,
        "votingDetails": [{
            "voter": "",
            "vote": "for|against|abstain",
            "indicators": [""]
        }],
        "totalVotes": {
            "for": number,
            "against": number,
            "abstain": number
        },
        "indicators": [""]
    }],
    "other": []
}

Note: "indicators" fields should contain exact phrases or keywords from the transcript that signal the corresponding event or information. Additionally, if there were no motions or votes, leave the respective lists empty.\n\n'''

def get_transcripts_from_db() -> list:
    """
    Fetch all meeting records from the database.
    Returns list of tuples: (meeting_id, transcript, location_name, date, title)
    """
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT meeting_id, transcript
        FROM meetings 
        WHERE transcript IS NOT NULL
    ''')
    
    meetings = cursor.fetchall()
    conn.close()
    return meetings

def init_analysis_tables():
    """Initialize the analysis database tables with proper schema"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meeting_analysis (
            meeting_id TEXT PRIMARY KEY,
            analysis_json TEXT,
            FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            name TEXT,
            speakers TEXT,  -- JSON array
            indicators TEXT,  -- JSON array
            FOREIGN KEY(meeting_id) REFERENCES meeting_analysis(meeting_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            name TEXT,
            did_pass BOOLEAN,
            votes_for INTEGER,
            votes_against INTEGER,
            votes_abstain INTEGER,
            indicators TEXT,  -- JSON array
            FOREIGN KEY(meeting_id) REFERENCES meeting_analysis(meeting_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voting_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vote_id INTEGER,
            voter TEXT,
            vote TEXT CHECK(vote IN ('for', 'against', 'abstain')),
            indicators TEXT,  -- JSON array
            FOREIGN KEY(vote_id) REFERENCES votes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vote_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator TEXT UNIQUE,
            frequency INTEGER DEFAULT 1,
            weight INTEGER DEFAULT 1,
            source_meetings TEXT  -- JSON array of meeting_ids
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_errors (
            meeting_id TEXT PRIMARY KEY,
            gpt_response TEXT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def update_vote_indicators(cursor, indicators: list, meeting_id: str):
    """Update the vote_indicators table with new indicators"""
    for indicator in indicators:
        # First try to get existing source_meetings
        cursor.execute('''
            SELECT source_meetings FROM vote_indicators 
            WHERE indicator = ?
        ''', (indicator,))
        result = cursor.fetchone()
        
        if result:
            # If indicator exists, update by parsing and modifying the JSON array
            existing_meetings = json.loads(result[0])
            if meeting_id not in existing_meetings:
                existing_meetings.append(meeting_id)
            
            cursor.execute('''
                UPDATE vote_indicators 
                SET frequency = frequency + 1,
                    source_meetings = ?
                WHERE indicator = ?
            ''', (json.dumps(existing_meetings), indicator))
        else:
            # If indicator doesn't exist, insert new record
            cursor.execute('''
                INSERT INTO vote_indicators (indicator, source_meetings)
                VALUES (?, ?)
            ''', (indicator, json.dumps([meeting_id])))

def process_meeting_transcript(transcript: str) -> Dict[Any, Any]:
    """
    Process a meeting transcript using GPT-4 to extract structured information.
    """
    client = openai.OpenAI()
    
    full_prompt = prompt + f"\n\nTranscript:\n{transcript}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            meeting_data = json.loads(response.choices[0].message.content)
            return meeting_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT response: {e}")
            print("Raw response:", response.choices[0].message.content)
            return None
            
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None

def save_analysis_to_db(meeting_id: str, analysis: Dict[Any, Any]):
    """Save the GPT analysis to structured database tables"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    try:
        # Save full analysis JSON
        cursor.execute('''
            INSERT OR REPLACE INTO meeting_analysis 
            (meeting_id, analysis_json)
            VALUES (?, ?)
        ''', (meeting_id, json.dumps(analysis)))
        
        # Save topics
        for topic in analysis['topics']:
            cursor.execute('''
                INSERT INTO topics 
                (meeting_id, name, speakers, indicators)
                VALUES (?, ?, ?, ?)
            ''', (
                meeting_id,
                topic['name'],
                json.dumps(topic['speakers']),
                json.dumps(topic['indicators'])
            ))
        
        # Save votes and their details
        for vote in analysis['votes']:
            cursor.execute('''
                INSERT INTO votes 
                (meeting_id, name, did_pass, 
                 votes_for, votes_against, votes_abstain, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                meeting_id,
                vote['name'],
                vote['didPass'],
                vote['totalVotes']['for'],
                vote['totalVotes']['against'],
                vote['totalVotes']['abstain'],
                json.dumps(vote['indicators'])
            ))
            vote_id = cursor.lastrowid

            # Update vote indicators
            update_vote_indicators(cursor, vote['indicators'], meeting_id)
            
            # Save only valid voting details (for, against, abstain)
            for detail in vote.get('votingDetails', []):
                if detail.get('vote') in ('for', 'against', 'abstain'):
                    cursor.execute('''
                        INSERT INTO voting_details 
                        (vote_id, voter, vote, indicators)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        vote_id,
                        detail['voter'],
                        detail['vote'],
                        json.dumps(detail.get('indicators', []))
                    ))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    
    finally:
        conn.close()

def get_last_processed_meeting():
    """Get the last processed meeting ID from the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT meeting_id FROM meeting_analysis 
        ORDER BY rowid DESC LIMIT 1
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def is_meeting_processed(meeting_id: str) -> bool:
    """Check if a meeting has already been processed"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 1 FROM meeting_analysis 
        WHERE meeting_id = ?
    ''', (meeting_id,))
    
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def log_processing_error(meeting_id: str, gpt_response: str, error_message: str):
    """Log an error that occurred during meeting processing"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO processing_errors 
            (meeting_id, gpt_response, error_message)
            VALUES (?, ?, ?)
        ''', (meeting_id, gpt_response, error_message))
        conn.commit()
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Process meeting transcripts')
    parser.add_argument('--resume', action='store_true', 
                       help='Skip meetings that have already been processed')
    args = parser.parse_args()

    # Initialize the database tables first
    init_analysis_tables()
    
    meetings = get_transcripts_from_db()
    total_meetings = len(meetings)
    print(f"Found {total_meetings} meetings in total")
    
    # If resuming, filter out already processed meetings
    if args.resume:
        unprocessed_meetings = [
            (mid, transcript) for mid, transcript in meetings 
            if not is_meeting_processed(mid)
        ]
        skipped = total_meetings - len(unprocessed_meetings)
        print(f"Skipping {skipped} already processed meetings")
        print(f"Processing {len(unprocessed_meetings)} remaining meetings")
        meetings = unprocessed_meetings
    
    # Create progress bar
    pbar = tqdm(meetings, desc="Processing meetings", unit="meeting")
    
    for meeting_id, transcript in pbar:
        pbar.set_description(f"Processing: {meeting_id}")
        
        try:
            meeting_data = process_meeting_transcript(transcript)
            
            if meeting_data:
                try:
                    save_analysis_to_db(meeting_id, meeting_data)
                except Exception as e:
                    error_msg = f"Database error: {str(e)}"
                    pbar.write(f"✗ {error_msg}")
                    log_processing_error(
                        meeting_id, 
                        json.dumps(meeting_data), 
                        error_msg
                    )
            else:
                error_msg = "Failed to analyze transcript"
                pbar.write(f"✗ {error_msg}")
                log_processing_error(meeting_id, "", error_msg)
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            pbar.write(f"✗ {error_msg}")
            log_processing_error(meeting_id, "", error_msg)
            continue
    
    pbar.close()

if __name__ == "__main__":
    main()