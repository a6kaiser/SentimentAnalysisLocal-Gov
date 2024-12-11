This is a two-day coding challenge to track people and their preferences using sentiment analysis on Whisper transcriptions.

Data structure:
- 'data.json' contains the names of transcription files to be processed, as well as some metadata about the video source
- transcripts (folder) contains the transcription files

Tasks
- Identify each government official
    - Label each with their position
    - Sentiments toward list of topics
- Create database of each of their voting records and passages related to each topic
- Inlcude sources for each passage

Methods:
- Scrapers:
    - scrape gov sites for names and positions
- Segmentation:
    - diarize audio to get speaker segments (not enough compute)
    - segment meetings into topics by keywords
- Labelers:
    - keyword indicators for each topic
    - keyword indicators for each vote
- Data cleaning:
    - voting records
    - unified discussions (ideally diarized) on a topic
- Predictive models:
    - likely topic boundary
    - likelihood of a vote passing
        - knn embedding method
        - EWM method with keyword experts
- Visualize:
    - Transcripts
    - Voting records
    - Topics
    - Sentiments
    - Keyword indicators
    - Predictive models

SQL Database schemas:
- meetings:
    - meeting_id
    - source
    - location_name
    - location_state
    - base_url
    - date
    - title
    - transcript
    - truncated (BOOL)
    - length_seconds
- meeting_analysis:
    - meeting_id (PK, FK -> meetings)
    - analysis_json
- topics:
    - id (PK)
    - meeting_id (FK -> meeting_analysis)
    - name
    - speakers (JSON array)
    - indicators (JSON array)
- votes:
    - id (PK)
    - meeting_id (FK -> meeting_analysis)
    - name
    - did_pass
    - votes_for
    - votes_against
    - votes_abstain
- voting_details:
    - id (PK)
    - vote_id (FK -> votes)
    - voter
    - vote ('for', 'against', 'abstain')
- vote_indicators:
    - id (PK)
    - indicator (unique)
    - frequency
    - weight
    - source_meetings (JSON array)
- processing_errors:
    - meeting_id (PK, FK -> meetings)
    - gpt_response
    - error_message
    - timestamp
- council_members:
    - id (PK)
    - location_name
    - location_state
    - member_name
    - position
    - UNIQUE(location_name, location_state, member_name)
- topic_boundary_labels:
    - meeting_id (PK, FK -> meetings)
    - char_index (PK)
    - label
    - context
    - PRIMARY KEY (meeting_id, char_index)

Vector Databases:
- meeting embeddings (not enough compute)
- snippet embeddings

Notes about the data:
- 1006 meetings transcribed (15 of which are truncated)
- 39 days 8 hours of audio transcription
- 10 locations
- 107 council members
- 24,304 thank you's (avg less than 2min apart)
- the sheer amount of stuttering and stammering is wild (hopefully a whisper issue but I doubt it)
- 98.1% of votes passed

Considerations:
- SBERT embedding time O(n^2)
- Keywords
    - thank you
    - my name is
    - ordinance
    - resolution
    - motion
    - vote
    - bill
    - amendment

Advanced methods for future consideration:
    - Transformer decoder for voting prediction
    - Self-attention decoder for topic segmentation


Pipeline:
- received meetings data
- db-transcripts -> sql db of meetings
- scrape-council_members -> sql db of council members
- label_gpt-meetings -> sql db of meeting topics, db of meeting votes, db of keyword features
- label-topic_boundaries(predict-topic_boundaries) -> label topic boundaries with a human in the loop
- db-topic_embeddings -> vector db of topic embeddings
- predict-voting(topic prompt) -> likelihood of vote passing
