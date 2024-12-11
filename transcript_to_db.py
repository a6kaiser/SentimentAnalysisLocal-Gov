import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def create_database():
    """Create the SQLite database and tables."""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    # Drop the table if it exists to start fresh
    cursor.execute('DROP TABLE IF EXISTS meetings')
    
    # Create the table with all columns
    cursor.execute('''
    CREATE TABLE meetings (
        meeting_id TEXT PRIMARY KEY,
        source TEXT,
        location_name TEXT,
        location_state TEXT,
        base_url TEXT,
        date DATE,
        title TEXT,
        transcript TEXT,
        truncated BOOLEAN,
        length_seconds INTEGER
    )
    ''')
    
    conn.commit()
    return conn

def load_transcript(transcript_path):
    """Load and stitch together transcript text from JSON file."""
    # Add counter dictionary as a function attribute if it doesn't exist
    if not hasattr(load_transcript, 'repair_counts'):
        load_transcript.repair_counts = {
            'normal_load': 0,
            'truncated_repair': 0
        }

    was_corrupted = False
    
    try:
        # First try normal loading
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            load_transcript.repair_counts['normal_load'] += 1
    except json.JSONDecodeError as e:
        was_corrupted = True
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Find the last valid closing brace and truncate everything after it
            last_brace_index = content.rfind('}')
            if last_brace_index != -1:
                truncated_content = content[:last_brace_index + 1] + ']'
                try:
                    transcript_data = json.loads(truncated_content)
                    load_transcript.repair_counts['truncated_repair'] += 1
                    tqdm.write(f"Successfully repaired JSON in {transcript_path} by truncating")
                except json.JSONDecodeError:
                    tqdm.write(f"Error: Could not repair JSON file at {transcript_path}")
                    return None, was_corrupted, 0
            else:
                tqdm.write(f"Error: No valid JSON structure found in {transcript_path}")
                return None, was_corrupted, 0
                
        except Exception as e:
            tqdm.write(f"Error: Failed to process file at {transcript_path}")
            tqdm.write(f"Error details: {str(e)}")
            return None, was_corrupted, 0
    
    # Only process if we successfully loaded the data
    if transcript_data:
        # Filter out segments without 'start' field and sort remaining ones
        valid_segments = [segment for segment in transcript_data if 'start' in segment]
        if len(valid_segments) < len(transcript_data):
            was_corrupted = True
            tqdm.write(f"Warning: Filtered out {len(transcript_data) - len(valid_segments)} invalid segments in {transcript_path}")
        
        valid_segments.sort(key=lambda x: x['start'])
        # Remove "uh" and "um" from the transcript text
        full_transcript = ' '.join(
            segment['text'].replace('&nbsp;', ' ')
                         .replace(' uh ', ' ')
                         .replace(' um ', ' ')
                         .replace(' Uh ', '')
                         .replace(' Um ', '')
            for segment in valid_segments
        )
        
        # Calculate length from last valid 'end' time
        length_seconds = 0
        segments_with_end = [s for s in valid_segments if 'end' in s and s['end'] is not None]
        if segments_with_end:
            length_seconds = max(segment['end'] for segment in segments_with_end)
        
        return full_transcript, was_corrupted, length_seconds
    return None, was_corrupted, 0

def process_meetings(data_path, transcripts_dir):
    """Process meetings data and transcripts."""
    conn = create_database()
    cursor = conn.cursor()
    
    # Add counters
    total_meetings = 0
    missing_transcripts = 0
    corrupted_transcripts = 0
    successful_transcripts = 0
    skipped_errors = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for org in tqdm(data, desc="Processing organizations"):
        org_location = org.get('location', {})
        org_base_url = org.get('base_url')
        
        for link_info in tqdm(org['link_infos'], 
                            desc=f"Processing meetings for {org_location.get('name', 'Unknown')}",
                            leave=False):
            total_meetings += 1
            
            # Skip if meeting has an error
            if 'error' in link_info:
                tqdm.write(f"Skipping meeting {link_info.get('meeting_id', 'unknown')} due to error in data.json")
                skipped_errors += 1
                continue
                
            meeting_id = link_info['meeting_id']
            transcript_path = Path(transcripts_dir) / f"{meeting_id}.json"
            
            if not transcript_path.exists():
                tqdm.write(f"Warning: No transcript found for meeting {meeting_id}")
                missing_transcripts += 1
                continue
            
            transcript_result = load_transcript(transcript_path)
            if transcript_result[0] is None:
                tqdm.write(f"Skipping meeting {meeting_id} due to transcript loading error")
                corrupted_transcripts += 1
                continue
            
            transcript_text, was_corrupted, length_seconds = transcript_result
            successful_transcripts += 1
            # Use meeting-specific location/base_url if available, fall back to org-level
            location = link_info.get('location', org_location)
            base_url = link_info.get('base_url', org_base_url)
            
            # Parse and format the date
            try:
                date_obj = datetime.strptime(link_info['date'], '%Y-%m-%d').date()
            except ValueError as e:
                print(f"Warning: Invalid date format for meeting {meeting_id}: {link_info['date']}")
                continue
            
            # Insert meeting data and transcript into database
            cursor.execute('''
            INSERT OR REPLACE INTO meetings (
                meeting_id,
                source,
                location_name,
                location_state,
                base_url,
                date,
                title,
                transcript,
                truncated,
                length_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meeting_id,
                link_info['source'],
                location.get('name'),
                location.get('state'),
                base_url,
                date_obj.isoformat(),
                link_info['title'],
                transcript_text,
                was_corrupted,
                length_seconds
            ))
    
    # Print statistics at the end
    tqdm.write("\nTranscript Processing Statistics:")
    tqdm.write(f"Total meetings processed: {total_meetings}")
    tqdm.write(f"Successfully processed transcripts: {successful_transcripts}")
    tqdm.write(f"Missing transcripts: {missing_transcripts}")
    tqdm.write(f"Completely corrupted transcripts: {corrupted_transcripts}")
    tqdm.write(f"Skipped due to errors in data.json: {skipped_errors}")
    tqdm.write(f"Success rate: {(successful_transcripts/total_meetings)*100:.2f}%")
    
    # Print repair statistics
    tqdm.write("\nRepair Pattern Statistics:")
    for pattern, count in load_transcript.repair_counts.items():
        tqdm.write(f"{pattern}: {count}")
    
    conn.commit()
    conn.close()

def main():
    # Configure paths
    data_path = 'data.json'
    transcripts_dir = 'transcripts'
    
    # Process meetings
    process_meetings(data_path, transcripts_dir)
    print("Database population complete!")

if __name__ == "__main__":
    main()
