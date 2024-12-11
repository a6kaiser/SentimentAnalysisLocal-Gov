from pyannote.audio import Pipeline
import torch
import json
from typing import Dict, List, Tuple
import wave
from pydub import AudioSegment
import sys
from datetime import timedelta
import os
import yt_dlp

class SpeakerDiarization:
    MAX_DURATION_SECONDS = 21 * 60  # 21 minutes in seconds
    
    def __init__(self, auth_token: str):
        """
        Initialize the diarization pipeline.
        auth_token: HuggingFace token with access to pyannote/speaker-diarization
        """
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=auth_token
        )
        
    def download_youtube_audio(self, url: str, output_dir: str = "downloads") -> str:
        """
        Download audio from YouTube video if it's under MAX_DURATION_SECONDS
        Returns path to downloaded audio file or raises Exception if video is too long
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # First, check video duration without downloading
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            
            if duration > self.MAX_DURATION_SECONDS:
                raise ValueError(
                    f"Video is too long: {timedelta(seconds=duration)} "
                    f"(max: {timedelta(seconds=self.MAX_DURATION_SECONDS)})"
                )
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            # macOS-specific options for better error handling
            'prefer_ffmpeg': True,
            'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'  # Common Homebrew path
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = os.path.join(output_dir, f"{info['id']}.wav")
                return audio_path
        except Exception as e:
            if "ffmpeg not found" in str(e):
                print("Error: FFmpeg not found. Please install it using: brew install ffmpeg")
                sys.exit(1)
            raise

    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Process audio file and return speaker segments
        """
        # Run diarization
        diarization = self.pipeline(audio_path)
        
        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        return segments

    def merge_with_whisper_transcript(self, 
                                    segments: List[Dict], 
                                    whisper_result: Dict) -> Dict:
        """
        Merge diarization results with Whisper transcript
        """
        enhanced_transcript = []
        
        for segment in whisper_result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            # Find matching speaker(s) for this time segment
            current_speakers = set()
            for diar_segment in segments:
                if (diar_segment["start"] <= end_time and 
                    diar_segment["end"] >= start_time):
                    current_speakers.add(diar_segment["speaker"])
            
            # Add speaker information to segment
            enhanced_segment = {
                "start": start_time,
                "end": end_time,
                "text": text,
                "speakers": list(current_speakers)
            }
            enhanced_transcript.append(enhanced_segment)
            
        return {"segments": enhanced_transcript}

def main():
    AUTH_TOKEN = "your_huggingface_token_here"
    
    # Initialize diarization
    diarizer = SpeakerDiarization(AUTH_TOKEN)
    
    # Example usage with YouTube URL
    youtube_url = "your_youtube_url_here"
    whisper_result = json.load(open("path_to_whisper_transcript.json"))
    
    try:
        # Download audio and get speaker segments
        audio_path = diarizer.download_youtube_audio(youtube_url)
        speaker_segments = diarizer.process_audio(audio_path)
        
        # Merge with Whisper transcript
        enhanced_transcript = diarizer.merge_with_whisper_transcript(
            speaker_segments, 
            whisper_result
        )
        
        # Save enhanced transcript
        with open("enhanced_transcript.json", "w") as f:
            json.dump(enhanced_transcript, f, indent=2)
            
        # Cleanup downloaded audio
        os.remove(audio_path)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
