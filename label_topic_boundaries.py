import sqlite3
from topic_boundaries import TopicSegmenter
from typing import List, Tuple
import os

class TopicBoundaryLabeler:
    def __init__(self):
        self.conn = sqlite3.connect('meetings.db')  # Updated to match your DB name
        self.cursor = self.conn.cursor()
        self.segmenter = TopicSegmenter()
        
        # Simplified table structure
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_boundary_labels (
                meeting_id TEXT,
                char_index INTEGER,
                label BOOLEAN,
                context TEXT,
                PRIMARY KEY (meeting_id, char_index),
                FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
            )
        """)
        self.conn.commit()

    def get_unlabeled_meetings(self) -> List[Tuple]:
        """Get meetings that haven't been fully labeled yet"""
        return self.cursor.execute("""
            SELECT DISTINCT m.meeting_id, m.transcript 
            FROM meetings m 
            LEFT JOIN topic_boundary_labels l ON m.meeting_id = l.meeting_id
            WHERE l.meeting_id IS NULL
            AND m.transcript IS NOT NULL
            AND m.corrupted = FALSE
            ORDER BY m.date DESC
        """).fetchall()

    def save_label(self, meeting_id: str, char_index: int, label: bool, context: str):
        """Save a single boundary label to the database"""
        self.cursor.execute("""
            INSERT INTO topic_boundary_labels 
                (meeting_id, char_index, label, context)
            VALUES (?, ?, ?, ?)
        """, (meeting_id, char_index, label, context))
        self.conn.commit()

    def display_boundary_context(self, text: str, boundary_index: int, context_chars: int = 200) -> str:
        """Show text before and after the boundary point"""
        start = max(0, boundary_index - context_chars)
        end = min(len(text), boundary_index + context_chars)
        
        # Get context before and after
        before = text[start:boundary_index]
        after = text[boundary_index:end]
        
        # Format display with clear separation
        display = (
            "\n" + "="*80 + "\n"
            f"BEFORE: ...{before}\n"
            "-"*40 + " BOUNDARY " + "-"*40 + "\n"
            f"AFTER: {after}...\n"
            "="*80 + "\n"
        )
        return display

    def get_boundary_indices(self, transcript: str) -> List[int]:
        """Extract character indices of all potential topic boundaries in a transcript."""
        # Get segments
        segments = self.segmenter.split_into_segments(transcript)
        boundary_indices = []
        
        # Track character position
        current_position = 0
        
        # Check each segment for topic starters
        for i, segment in enumerate(segments):
            if self.segmenter.is_topic_starter(segment):
                # Calculate character index for this boundary
                boundary_indices.append(current_position)
            
            # Update position counter (add segment length plus space)
            current_position += len(segment) + 1  # +1 for the space between segments
            
        return boundary_indices

    def run_labeling_session(self):
        """Main labeling loop"""
        meetings = self.get_unlabeled_meetings()
        
        print(f"Found {len(meetings)} meetings to label")
        for meeting_id, transcript in meetings:
            print(f"\nProcessing meeting: {meeting_id}")
            
            # Get potential boundaries
            boundary_indices = self.get_boundary_indices(transcript)
            
            if not boundary_indices:
                print("No potential boundaries found in this meeting")
                continue
            
            print(f"Found {len(boundary_indices)} potential boundaries to label")
            
            for boundary_idx in boundary_indices:
                context = self.display_boundary_context(transcript, boundary_idx)
                print(context)
                
                while True:
                    response = input("Is this a valid topic boundary? (y/n/q to quit): ").lower()
                    
                    if response == 'q':
                        print("Saving progress and quitting...")
                        self.conn.close()
                        return
                    elif response in ['y', 'n']:
                        self.save_label(
                            meeting_id=meeting_id,
                            char_index=boundary_idx,
                            label=(response == 'y'),
                            context=context
                        )
                        break
                    else:
                        print("Please enter 'y', 'n', or 'q'")
            
            print(f"Completed labeling for meeting {meeting_id}")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    labeler = TopicBoundaryLabeler()
    labeler.run_labeling_session()

if __name__ == "__main__":
    main()
