import re
import json
import logging
from typing import Optional, List, Dict
from html import unescape
import requests
from yt_dlp import YoutubeDL

# filepath: c:\Gituhb\AI\AIVidCreator-main\content_extractors\tiktok_transcript.py


"""
TikTok Transcript Extractor
Tries to extract captions/subtitles (VTT/SRT/JSON) from a TikTok video page.
Provides:
 - extract_video_id(url) -> Optional[str]
 - get_transcript(video_url, languages=['en', ...]) -> Optional[str]
"""

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract numeric TikTok video ID from common URL formats.
    Returns None if not found.
    """
    patterns = [
        # https://www.tiktok.com/@user/video/12345
        r'tiktok\.com/.+?/video/(\d+)',
        r'tiktok\.com/video/(\d+)',      # alternative
        # short link (can't get id directly)
        r'vm\.tiktok\.com/([A-Za-z0-9]+)',
        r'tt\.ly/([A-Za-z0-9]+)',         # other shortener
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _fetch_page(url: str, timeout: int = 10) -> Optional[requests.Response]:
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers,
                            timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.debug(f"Failed to fetch page {url}: {e}")
        return None


def _find_caption_urls_in_html(html: str) -> List[str]:
    """
    Heuristics to find caption/subtitle file URLs in the page HTML.
    Looks for .vtt/.srt and escaped JSON fields like "captionUrl".
    """
    urls = set()

    # 1) Simple direct .vtt or .srt links in the HTML
    for m in re.finditer(r'https?://[^\s"\'<>]+?\.(?:vtt|srt)(?:\?[^"\']*)?', html, re.IGNORECASE):
        urls.add(unescape(m.group(0)))

    # 2) JSON-ish fields like "captionUrl":"https:\/\/..."
    for m in re.finditer(r'"captionUrl"\s*:\s*"([^"]+)"', html):
        urls.add(unescape(m.group(1).replace('\\/', '/')))

    # 3) Some pages include "subtitles": [{"url":"..."}]
    for m in re.finditer(r'"subtitles"\s*:\s*(\[[^\]]+\])', html):
        try:
            arr = json.loads(m.group(1))
            for item in arr:
                if isinstance(item, dict) and 'url' in item:
                    urls.add(item['url'])
        except Exception:
            continue

    # 4) Generic src attributes that reference subtitle files
    for m in re.finditer(r'src=["\'](https?://[^"\']+?\.(?:vtt|srt)(?:\?[^"\']*)?)["\']', html, re.IGNORECASE):
        urls.add(unescape(m.group(1)))

    return list(urls)


def _parse_vtt(text: str) -> List[Dict]:
    """
    Parse simple WebVTT content into list of entries:
    {'start': float, 'end': float, 'duration': float, 'text': str}
    """
    entries: List[Dict] = []
    lines = text.splitlines()
    i = 0
    timestamp_re = re.compile(
        r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{2})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{2})')

    def _to_seconds(t: str) -> float:
        parts = t.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float(parts[0])

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # skip numeric index if present
        if line.isdigit():
            i += 1
            line = lines[i].strip() if i < len(lines) else ''
        m = timestamp_re.search(line)
        if m:
            start_s = _to_seconds(m.group(1))
            end_s = _to_seconds(m.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            entry_text = ' '.join(text_lines).strip()
            entries.append({
                'start': start_s,
                'end': end_s,
                'duration': max(0.0, end_s - start_s),
                'text': entry_text
            })
        else:
            i += 1
    return entries


def _parse_simple_srt(text: str) -> List[Dict]:
    # re-use vtt parser logic since formats are similar
    return _parse_vtt(text)


def _fetch_and_parse_subtitle(url: str) -> Optional[List[Dict]]:
    resp = _fetch_page(url)
    if not resp:
        return None
    content_type = resp.headers.get('Content-Type', '').lower()
    body = resp.text
    # If JSON structured captions
    try:
        j = json.loads(body)
        # try common shapes
        # Example: {"captions": [{"text":"...","start":0,"dur":2}, ...]}
        if isinstance(j, dict):
            if 'captions' in j and isinstance(j['captions'], list):
                parsed = []
                for c in j['captions']:
                    text = c.get('text') or c.get('content') or ''
                    start = float(c.get('start', 0))
                    dur = float(c.get('duration', c.get('dur', 0)))
                    parsed.append({
                        'start': start,
                        'end': start + dur,
                        'duration': dur,
                        'text': text
                    })
                return parsed
            # Some TikTok caption endpoints return "data": [{"text":"...","from":..,"to":..}, ...]
            if 'data' in j and isinstance(j['data'], list):
                parsed = []
                for c in j['data']:
                    text = c.get('text') or ''
                    start = float(c.get('from', 0))
                    end = float(c.get('to', start))
                    parsed.append({'start': start, 'end': end, 'duration': max(
                        0.0, end - start), 'text': text})
                if parsed:
                    return parsed
    except Exception:
        pass

    # If VTT or SRT
    if '.vtt' in url or 'text/vtt' in content_type or body.strip().startswith('WEBVTT'):
        return _parse_vtt(body)
    if '.srt' in url or 'application/x-subrip' in content_type or re.search(r'^\d+\s*?\n', body):
        return _parse_simple_srt(body)

    # As fallback, attempt to extract plain text-like subtitles by stripping timestamps
    # split by newlines and return as sequential items without timestamps
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        parsed = []
        time = 0.0
        step = 2.0
        for ln in lines:
            parsed.append({'start': time, 'end': time + step,
                          'duration': step, 'text': ln})
            time += step
        return parsed

    return None


def get_transcript(video_url: str, languages: List[str] = ['en']) -> Optional[str]:
    """
    Get transcript text from a TikTok video URL.
    Tries to find subtitle/caption files (VTT/SRT/JSON) on the video page.
    Returns a single joined string of captions, or None if not found.
    """
    logger.debug(f"Fetching TikTok transcript for: {video_url}")
    # Try yt-dlp first (may have subtitles cached or accessible)
    try:
        ydl_opts = {'skip_download': True,
                    'writesubtitles': True, 'writeautomaticsub': True}
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            # yt-dlp may include subtitles in info dict
            subs = info.get('subtitles') or info.get(
                'requested_subtitles') or info.get('automatic_captions')
            if subs and isinstance(subs, dict):
                # pick first available language
                for lang, meta in subs.items():
                    try:
                        if not isinstance(lang, str):
                            continue
                        if lang.lower() not in ('eng-us', 'heb-il', 'en'):
                            continue
                        # meta might be a list of dicts with 'url' or text content
                        if isinstance(meta, list) and len(meta) > 0:
                            item = meta[0]
                            if isinstance(item, dict) and 'url' in item:
                                url = item['url']
                                parsed = _fetch_and_parse_subtitle(url)
                                if parsed:
                                    full_text = ' '.join(
                                        entry.get('text', '') for entry in parsed).strip()
                                    if full_text:
                                        return full_text
                            if isinstance(item, str):
                                text = item.strip()
                                if text:
                                    return text
                    except Exception:
                        continue
    except Exception:
        pass

    resp = _fetch_page(video_url)
    if not resp:
        logger.error("Failed to fetch video page")
        return None

    html = resp.text
    # attempt to locate caption URLs
    caption_urls = _find_caption_urls_in_html(html)

    # If no direct caption links, attempt to find escaped JSON blocks that may contain caption fields
    if not caption_urls:
        # look for "captionUrl" with escaped slashes
        m = re.search(r'"captionUrl"\s*:\s*"([^"]+)"', html)
        if m:
            caption_urls.append(unescape(m.group(1).replace('\\/', '/')))

    # If still empty, try to find any .vtt in the entire HTML (looser search)
    if not caption_urls:
        loose = re.findall(r'https?://[^\s"\'<>]+?\.vtt[^\s"\'<>]*', html)
        for u in loose:
            caption_urls.append(unescape(u))

    # Try each found caption URL (prefer language-aware URLs if present)
    for url in caption_urls:
        logger.debug(f"Trying caption URL: {url}")
        parsed = _fetch_and_parse_subtitle(url)
        if parsed:
            # join into single string
            full_text = ' '.join(entry.get('text', '')
                                 for entry in parsed).strip()
            if full_text:
                logger.debug(f"Found transcript length: {len(full_text)}")
                return full_text

    # Last resort: try to extract text from embedded JSON ItemModule description fields (may contain spoken text)
    try:
        # get ItemModule JSON if present
        m = re.search(
            r'<script id="SIGI_STATE"[^>]*>(.*?)</script>', html, re.S)
        if m:
            try:
                sigi = json.loads(m.group(1))
                item_module = sigi.get('ItemModule') or {}
                for key, item in item_module.items():
                    # description text is often in 'desc'
                    desc = item.get('desc')
                    if desc and isinstance(desc, str):
                        # return description as fallback (not real transcript)
                        return desc.strip()
            except Exception:
                pass
    except Exception:
        pass

    logger.info("No captions found on TikTok page")
    return None


# if __name__ == "__main__":
#     # Example usage
#     test_url = "https://www.tiktok.com/@jayshetty/video/7498778765581896991?is_from_webapp=1&sender_device=pc"
#     print("Fetching transcript...")
#     txt = get_transcript(test_url, ['eng-us'])
#     if txt:
#         print(f"Transcript preview: {txt[:400]}")
#     else:
#         print("No transcript found.")
