import re
import spacy
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class TopicSegmenter:
    def __init__(self):
        # Initialize spaCy for better text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Comprehensive patterns for topic boundaries
        self.topic_starters = [
            # Agenda items and numbers
            r"(?i)(?:agenda\s+)?item\s+(?:number\s+)?(?:#?\d+|[a-z])",
            r"(?i)number\s+\d+",
            
            # Specific document types
            r"(?i)ordinance\s+(?:number\s+)?#?\d+[-\w]*",
            r"(?i)resolution\s+(?:number\s+)?#?\d+[-\w]*",
            r"(?i)bill\s+(?:number\s+)?#?\d+[-\w]*",
            
            # Meeting segments
            r"(?i)public\s+hearing",
            r"(?i)public\s+comment",
            r"(?i)citizen\s+comments?",
            r"(?i)old\s+business",
            r"(?i)new\s+business",
            
            # Transitional phrases
            r"(?i)moving\s+(?:on\s+)?to",
            r"(?i)next\s+(?:item|order\s+of\s+business)",
            r"(?i)let's\s+move\s+to",
            r"(?i)turning\s+(?:our\s+attention\s+)?to",
            
            # Action items
            r"(?i)consideration\s+of",
            r"(?i)discussion\s+(?:regarding|concerning|about)",
            r"(?i)presentation\s+(?:on|regarding|about)",
        ]
        
        self.topic_enders = [
            # Voting results
            r"(?i)motion\s+(?:carries|passed|approved|denied|fails)",
            r"(?i)vote\s+results?:?",
            r"(?i)the\s+(?:motion|resolution|ordinance)\s+(?:is|was)\s+(?:approved|passed|adopted|denied)",
            r"(?i)all\s+(?:those\s+)?in\s+favor\s+say\s+aye",
            r"(?i)the\s+ayes\s+have\s+it",
            
            # Transitions
            r"(?i)moving\s+on\s+to\s+(?:the\s+)?next",
            r"(?i)that\s+concludes",
            r"(?i)next\s+item",
            
            # Meeting segments
            r"(?i)this\s+concludes\s+(?:the|our)",
            r"(?i)end\s+of\s+(?:discussion|presentation)",
        ]
        
        self.motion_patterns = [
            r"(?i)(?P<mover>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+moves?",
            r"(?i)motion\s+by\s+(?P<mover>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?i)moved\s+by\s+(?P<mover>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        
        self.second_patterns = [
            r"(?i)seconded\s+by\s+(?P<seconder>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?i)second\s+(?:by|from)\s+(?P<seconder>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        
        self.vote_patterns = [
            r"(?i)vote:\s*(?P<ayes>\d+)\s*ayes?,\s*(?P<nays>\d+)\s*nays?",
            r"(?i)(?P<ayes>\d+)\s*in\s*favor,\s*(?P<nays>\d+)\s*opposed",
            r"(?i)motion\s+(?P<result>carries|passes|fails|denied)",
        ]

    def split_into_segments(self, transcript: str) -> List[str]:
        """Split transcript into meaningful segments using sentence boundaries."""
        # Clean the transcript
        cleaned_text = re.sub(r'\s+', ' ', transcript).strip()
        
        # Use NLTK for initial sentence tokenization
        sentences = sent_tokenize(cleaned_text)
        
        # Further split long sentences at common boundary markers
        segments = []
        for sentence in sentences:
            # Split at common markers while preserving them
            parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z])', sentence)
            segments.extend(parts)
        
        return [s.strip() for s in segments if s.strip()]

    def is_topic_starter(self, segment: str) -> bool:
        """Check if segment contains a topic starter pattern."""
        return any(re.search(pattern, segment) for pattern in self.topic_starters)

    def is_topic_ender(self, segment: str) -> bool:
        """Check if segment contains a topic ender pattern."""
        return any(re.search(pattern, segment) for pattern in self.topic_enders)

    def extract_motion(self, segment: str) -> Dict:
        """Extract motion details from segment."""
        motion = {'text': segment, 'mover': None, 'seconder': None}
        
        # Extract mover
        for pattern in self.motion_patterns:
            match = re.search(pattern, segment)
            if match and 'mover' in match.groupdict():
                motion['mover'] = match.group('mover')
                break
        
        # Extract seconder
        for pattern in self.second_patterns:
            match = re.search(pattern, segment)
            if match and 'seconder' in match.groupdict():
                motion['seconder'] = match.group('seconder')
                break
        
        return motion

    def extract_vote(self, segment: str) -> Dict:
        """Extract vote details from segment."""
        vote = {'text': segment, 'ayes': None, 'nays': None, 'result': None}
        
        for pattern in self.vote_patterns:
            match = re.search(pattern, segment)
            if match:
                groupdict = match.groupdict()
                if 'ayes' in groupdict:
                    vote['ayes'] = int(groupdict['ayes'])
                if 'nays' in groupdict:
                    vote['nays'] = int(groupdict['nays'])
                if 'result' in groupdict:
                    vote['result'] = groupdict['result']
                break
        
        return vote

    def extract_topic_segments(self, transcript: str) -> List[Dict]:
        """Extract topic segments with their votes/motions."""
        segments = self.split_into_segments(transcript)
        topics = []
        current_topic = {
            'start_index': 0,
            'start_text': '',
            'text': '',
            'motions': [],
            'votes': [],
            'confidence': 0.0
        }
        
        for i, segment in enumerate(segments):
            # Check for new topic
            if self.is_topic_starter(segment):
                if current_topic['text']:
                    # Calculate confidence score for previous topic
                    current_topic['confidence'] = self._calculate_confidence(current_topic)
                    topics.append(current_topic)
                
                current_topic = {
                    'start_index': i,
                    'start_text': segment,
                    'text': segment,
                    'motions': [],
                    'votes': [],
                    'confidence': 0.0
                }
            else:
                current_topic['text'] += ' ' + segment
            
            # Check for motions and votes
            for pattern in self.motion_patterns:
                if re.search(pattern, segment):
                    current_topic['motions'].append(self.extract_motion(segment))
            
            for pattern in self.vote_patterns:
                if re.search(pattern, segment):
                    current_topic['votes'].append(self.extract_vote(segment))
            
            # If we find a topic ender, close the current topic
            if self.is_topic_ender(segment):
                current_topic['confidence'] = self._calculate_confidence(current_topic)
                topics.append(current_topic)
                current_topic = {
                    'start_index': i + 1,
                    'start_text': '',
                    'text': '',
                    'motions': [],
                    'votes': [],
                    'confidence': 0.0
                }
        
        # Add the last topic if it exists
        if current_topic['text']:
            current_topic['confidence'] = self._calculate_confidence(current_topic)
            topics.append(current_topic)
        
        return topics

    def _calculate_confidence(self, topic: Dict) -> float:
        """Calculate confidence score for topic segmentation."""
        confidence = 0.0
        
        # Higher confidence if we have a clear starter
        if self.is_topic_starter(topic['start_text']):
            confidence += 0.4
        
        # Higher confidence if we have motions and votes
        if topic['motions']:
            confidence += 0.3
        if topic['votes']:
            confidence += 0.3
        
        # Length-based confidence (penalize very short or very long segments)
        word_count = len(topic['text'].split())
        if 50 <= word_count <= 1000:
            confidence += 0.2
        elif word_count < 20 or word_count > 2000:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))

def identify_topic_boundaries(transcript: str) -> List[Dict]:
    """Main function to identify topic boundaries in a transcript."""
    segmenter = TopicSegmenter()
    return segmenter.extract_topic_segments(transcript)
