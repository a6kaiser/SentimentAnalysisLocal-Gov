import sqlite3
import re

def create_snippet_table(cursor):
    cursor.execute('DROP TABLE IF EXISTS transcript_snippets')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transcript_snippets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id TEXT,
        speaker_hypothesis TEXT,
        transcript_snippet TEXT,
        FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
    )''')

def find_names_from_introductions(text):
    # Pattern matches "I'm" or "I am" followed by sequence of capitalized words
    # Removed re.IGNORECASE flag to enforce capitalization
    pattern = r"(?:I am|I'm)\s+([A-Z][a-zA-Z\']*(?:[\s0-9\-\']+[A-Z][a-zA-Z\']*)*)\b"
    
    names = []
    # We only want case-insensitive matching for "I am"/"I'm", not for the name part
    matches = re.finditer(pattern.replace("I am|I'm", "(?i:I am|I'm)"), text)
    
    for match in matches:
        names.append(match.group(1))
    
    return names

def find_capitalized_words_before_thank_you(text):
    # Find the first occurrence of "thank you"
    match = re.search(r'thank you', text, re.IGNORECASE)
    if not match:
        return []
    
    # Get all text before the first "thank you"
    before_text = text[:match.start()]
    
    # Find sequences of capitalized words, allowing numbers, whitespace, and name punctuation
    names = re.findall(r'\b[A-Z][a-zA-Z\']*(?:[\s0-9\-\']+[A-Z][a-zA-Z\']*)*\b', before_text)
    return names

def split_on_thank_you(transcript):
    # Find all occurrences of "thank you"
    thank_you_positions = [(m.start(), m.end()) for m in re.finditer(r'thank you', transcript, flags=re.IGNORECASE)]
    
    snippets = []
    for i, (start, end) in enumerate(thank_you_positions):
        # Get approximately 15 words before "thank you"
        words_before = transcript[:start].split()
        context_start = 0
        if len(words_before) >= 15:
            context_start = len(' '.join(words_before[-15:]))
            try:
                context_start = transcript.rindex(' ', 0, start - context_start) + 1
            except ValueError:
                context_start = max(0, start - context_start)
        
        # Find the end (either next "thank you" or end of transcript)
        if i < len(thank_you_positions) - 1:
            context_end = thank_you_positions[i + 1][0] + len("thank you")
        else:
            context_end = len(transcript)
            
        snippet = transcript[context_start:context_end].strip()
        snippets.append(snippet)
    
    return snippets

def find_names_after_my_name_is(text):
    # Pattern matches "my name is" followed by a word and optionally more capitalized words
    pattern = r"my name is\s+([a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
    
    names = []
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        name_part = match.group(1)
        # Take the first word regardless of capitalization
        words = name_part.split()
        if len(words) > 1:
            # For additional words, only include if capitalized
            additional_words = [w for w in words[1:] if w[0].isupper()]
            name = ' '.join([words[0]] + additional_words)
        else:
            name = words[0]
        names.append(name)
    
    return names

def process_transcripts():
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    # Create new table
    create_snippet_table(cursor)
    
    cursor.execute('SELECT meeting_id, transcript FROM meetings')
    meetings = cursor.fetchall()
    
    for meeting_id, transcript in meetings:
        if not transcript:
            continue
            
        snippets = split_on_thank_you(transcript)
        
        for snippet in snippets:
            # Find potential speakers
            speakers = set()
            
            # Add names from introductions
            speakers.update(find_names_from_introductions(snippet))
            
            # Add names before "thank you"
            speakers.update(find_capitalized_words_before_thank_you(snippet))
            
            # Add names after "my name is"
            speakers.update(find_names_after_my_name_is(snippet))
            
            # Store the snippet
            speaker_hypothesis = ', '.join(speakers) if speakers else None
            
            cursor.execute('''
                INSERT INTO transcript_snippets (meeting_id, speaker_hypothesis, transcript_snippet)
                VALUES (?, ?, ?)
            ''', (meeting_id, speaker_hypothesis, snippet))
    
    conn.commit()
    conn.close()

def sample_thank_you_contexts(limit=5, context_chars=100):
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    # Get random samples from transcript_snippets
    cursor.execute('''
        SELECT transcript_snippet, speaker_hypothesis, meeting_id 
        FROM transcript_snippets 
        WHERE transcript_snippet LIKE '%thank you%' 
        ORDER BY RANDOM() 
        LIMIT ?
    ''', (limit,))
    
    samples = cursor.fetchall()
    
    print(f"\n=== Sampling {limit} 'thank you' contexts ===\n")
    for snippet, speakers, meeting_id in samples:       
        print("Meeting ID:", meeting_id)
        print("Context:", snippet)
        print("Hypothesized speakers:", speakers or "None detected")
        print("-" * 80 + "\n")
    
    conn.close()

if __name__ == "__main__":
    process_transcripts()
    sample_thank_you_contexts()
