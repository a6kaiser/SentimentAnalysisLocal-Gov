# Council Meeting Analysis Project 🎙️

A two-day coding challenge to track people and their preferences using sentiment analysis on Whisper transcriptions.

## 🎯 Project Achievements
- `scrape-council.py`: Scraped gov sites for names and positions
- `label_gpt-meetings.py`: Labeled meetings with topics and voting results
- `match-votes_topics`: Matched voting results to summary of the topic
- `predict-knn_voting.py`: Predicted voting results in each city based off topic prompt
- `rag-sentiment_search.py`: Sentiment analysis for each city based off topic prompt

## 🚧 Work in Progress
- **Predict topic boundaries**: Enable efficient boundary labeling through active learning
- **Diarize audio**: Speaker segmentation for improved analysis
- **EXP3 algorithm**: Voting prediction using topic clusters as arms

## 🔮 Future Considerations
- Finetune transformer for voting prediction
- Self-attention decoder for topic segmentation

## 📁 Data for the assignment
- `data.json`: Contains transcription files metadata
- `transcripts/`: Contains the transcription files

## 🎯 Tasks
### Primary Objectives
- Identify government officials
  - Position labeling
  - Topic sentiment analysis
- Create voting records database
- Source documentation

## 🛠��� Methods

### Scrapers
- Gov sites data extraction

### Segmentation
- Audio diarization (future work)
- Topic-based meeting segmentation

### Labelers
- Topic keyword indicators
- Vote keyword indicators

### Data Cleaning
- Voting records
- Topic discussions

### Predictive Models
- Topic boundary prediction
- Vote passing likelihood
  - KNN embedding
  - EWM with keyword experts

### Visualization
- Transcripts
- Voting records
- Topics
- Sentiments
- Keyword indicators
- Predictive models

## 💾 Database Schema

### SQL Tables
<details>
<summary>Click to expand database schemas</summary>

#### Core Tables
- **meetings**
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

- **meeting_analysis**
  - meeting_id (PK, FK -> meetings)
  - analysis_json

- **topics**
  - id (PK)
  - meeting_id (FK -> meeting_analysis)
  - name
  - speakers (JSON array)
  - indicators (JSON array)

#### Voting Related
- **votes**
  - id (PK)
  - meeting_id (FK -> meeting_analysis)
  - name
  - did_pass
  - votes_for
  - votes_against
  - votes_abstain

- **voting_details**
  - id (PK)
  - vote_id (FK -> votes)
  - voter
  - vote ('for', 'against', 'abstain')

- **vote_indicators**
  - id (PK)
  - indicator (unique)
  - frequency
  - weight
  - source_meetings (JSON array)

#### Members and Labels
- **council_members**
  - id (PK)
  - location_name
  - location_state
  - member_name
  - position
  - UNIQUE(location_name, location_state, member_name)

- **topic_boundary_labels**
  - meeting_id (PK, FK -> meetings)
  - char_index (PK)
  - label
  - context
  - PRIMARY KEY (meeting_id, char_index)

#### Error Tracking
- **processing_errors**
  - meeting_id (PK, FK -> meetings)
  - gpt_response
  - error_message
  - timestamp
</details>

### Vector Databases
- Meeting embeddings
- Snippet embeddings

## 📊 Data Insights
- 1006 meetings transcribed (15 truncated)
- 39 days 8 hours of audio
- 10 locations
- 107 council members
- 24,304 "thank you"s
- 98.1% vote pass rate

## 🔑 Key Considerations
### SBERT Embedding
- Time complexity: O(n²)

### Keywords
- thank you
- my name is
- ordinance
- resolution
[... other keywords ...]

## 🔄 Pipeline Flow
1. Receive meetings data
2. DB transcripts → SQL
3. Scrape council members
4. Label meetings (GPT)
5. Label topic boundaries
6. Generate topic embeddings
7. Predict voting
