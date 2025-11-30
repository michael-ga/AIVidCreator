import os
import yt_dlp
from typing import Optional
import pathlib
import shutil
import sys
import logging


ydl_audio_opts = {
    'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }]
}


def _make_outtmpl(target_file: str) -> str:
    # Ensure parent directory exists
    parent = os.path.dirname(target_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    base, ext = os.path.splitext(target_file)
    # use a template so yt-dlp can choose the initial extension; postprocessor
    # (FFmpegExtractAudio) will then create the final mp3 file from the temp file.
    return base + '.%(ext)s'


def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable.

    Search order:
      1. `FFMPEG_PATH` environment variable
      2. If running from a PyInstaller bundle, check next to the executable
      3. `third_party/ffmpeg/ffmpeg.exe` inside the repo
      4. System PATH via `shutil.which('ffmpeg')`
      5. A user OneDrive path (best-effort)

    Returns an absolute path or None.
    """
    # 1) Env override
    env_path = os.environ.get('FFMPEG_PATH')
    if env_path:
        if os.path.isfile(env_path):
            return os.path.abspath(env_path)

    # 2) PyInstaller bundle
    try:
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            candidate = os.path.join(exe_dir, 'ffmpeg.exe')
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    except Exception:
        pass

    # 3) Project third_party folder
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    candidate = repo_root / 'third_party' / 'ffmpeg' / 'ffmpeg.exe'
    if candidate.exists():
        return str(candidate)

    # 4) System PATH
    which_path = shutil.which('ffmpeg')
    if which_path:
        return which_path

    # 5) Common user OneDrive location (best-effort)
    possible = os.path.expandvars(
        r"C:\Users\mgabbay\OneDrive - Intel Corporation\Documents\Youtube\ffmpeg.exe")
    if os.path.isfile(possible):
        return possible

    logging.getLogger(__name__).debug('ffmpeg not found by any method')
    return None


def get_video(url: str, target_file: str) -> Optional[str]:
    opts = {}
    opts['noplaylist'] = True
    opts['outtmpl'] = _make_outtmpl(target_file)
    ff = _find_ffmpeg()
    if ff:
        opts['ffmpeg_location'] = ff
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        # best-effort: return the exact target_file (user likely chose full name)
        return target_file
    except Exception:
        return None


def get_sound(url: str, target_file: str) -> Optional[str]:
    """Download audio from `url` and save to `target_file`.

    Uses `ydl_audio_opts` and writes a temporary download to the same
    base path as `target_file` (yt-dlp will append the original media
    extension, e.g. .m4a, then FFmpegExtractAudio will produce .mp3).
    Returns the expected mp3 path on success or None.
    """
    opts = dict(ydl_audio_opts)
    opts['noplaylist'] = True
    opts['outtmpl'] = _make_outtmpl(target_file)
    ff = _find_ffmpeg()
    if ff:
        opts['ffmpeg_location'] = ff
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        # We used preferredcodec mp3 in the postprocessor so return .mp3 path
        base, _ = os.path.splitext(target_file)
        final = base + '.mp3'
        if os.path.exists(final):
            return final
        # fallback: return the requested path
        return target_file
    except Exception:
        return None


# if __name__ == '__main__':
#     # only run download when executed directly
#     get_sound(
#         URLS[0], 'C:\\Gituhb\\AI\\AIVidCreator-main\\content_extractors\\test')
