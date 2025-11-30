from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from typing import Optional, List, Dict
import logging

"""
YouTube Transcript Extractor
Extracts transcripts/captions from YouTube videos.
"""


logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.

    Args:
        url: YouTube URL (watch?v=, youtu.be/, embed/, etc.)

    Returns:
        Video ID or None if not found
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_transcript(video_url: str, languages: List[str] = ['en', 'heb']) -> Optional[Dict]:
    """
    Get transcript from a YouTube video.

    Args:
        video_url: YouTube video URL or ID
        languages: List of preferred language codes (default: ['en'])

    Returns:
        Dictionary with transcript data or None if unavailable
    """
    video_id = extract_video_id(video_url)
    print(f"[DEBUG] Extracted video_id: {video_id}")

    if not video_id:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        print(f"[ERROR] Could not extract video ID from URL: {video_url}")
        return None

    try:
        print(
            f"[DEBUG] Fetching transcript for video_id: {video_id} with languages: {languages}")
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        print(f"[DEBUG] Transcript list length: {len(transcript_list)}")

        # Use attribute access for transcript objects (compat with FetchedTranscriptSnippet)
        full_text = ' '.join([getattr(entry, 'text', str(entry))
                             for entry in transcript_list])
        print(f"[DEBUG] Full transcript length: {len(full_text)}")
        return full_text
        # return {
        #     'video_id': video_id,
        #     'video_url': f"https://www.youtube.com/watch?v={video_id}",
        #     'transcript': full_text,
        #     'raw_transcript': transcript_list,
        #     'language': languages[0]
        # }

    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video: {video_id}")
        print(f"[ERROR] Transcripts are disabled for video: {video_id}")
        return None
    except NoTranscriptFound:
        logger.error(
            f"No transcript found for video: {video_id} in languages: {languages}")
        print(
            f"[ERROR] No transcript found for video: {video_id} in languages: {languages}")
        return None
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        print(f"[ERROR] Exception: {str(e)}")
        return None


def get_transcript_with_timestamps(video_url: str, languages: List[str] = ['en']) -> Optional[List[Dict]]:
    """
    Get transcript with timestamps from a YouTube video.

    Args:
        video_url: YouTube video URL or ID
        languages: List of preferred language codes (default: ['en'])

    Returns:
        List of transcript entries with timestamps or None if unavailable
    """
    video_id = extract_video_id(video_url)

    if not video_id:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=languages
        )
        return transcript_list

    except Exception as e:
        logger.error(f"Error fetching transcript with timestamps: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    test_url = "https://www.youtube.com/watch?v=r2hMfdDpRrs"

    print("Fetching transcript...")
    result = get_transcript(test_url)

    if result:
        print(f"\nVideo ID: {result['video_id']}")
        print(f"Language: {result['language']}")
        print(
            f"\nTranscript Preview (first 500 chars): {result['transcript'][:500]}")
        print("\n...")
    else:
        print("Failed to fetch transcript")
