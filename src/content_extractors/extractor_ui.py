"""
Simple modern Tkinter UI for extracting audio, video and transcripts.

- URL entry
- Three buttons with icons: Extract Audio, Extract Video, Extract Transcript
- Hebrew menu labels
- Background threads so UI stays responsive

This module only launches the GUI when run as `__main__`.
"""
from __future__ import annotations
import os
import sys

import threading
import queue
import traceback
from urllib.parse import urlparse
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
# ensure imports work when running this module directly:
# add the current directory and its parent (project root) to sys.path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_here, _root):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
# Simple import handling: add project root to sys.path then try absolute
# import, fallback to local module import. Keeps behavior simple and robust
# when running the file directly or as part of the package.
try:
    # Ensure project root is on sys.path so absolute imports work when running
    # this module directly from the content_extractors folder.
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)
except Exception:
    pass

get_sound = None
get_video = None
try:
    from content_extractors.video_audio_downloader import get_sound, get_video
except Exception:
    try:
        # when executed from within the content_extractors directory
        from video_audio_downloader import get_sound, get_video
    except Exception:
        get_sound = None
        get_video = None

try:
    from content_extractors.youtube_transcript import get_transcript as yt_get_transcript
except Exception:
    yt_get_transcript = None

try:
    from content_extractors.tiktok_transcript import get_transcript as tiktok_get_transcript
except Exception:
    tiktok_get_transcript = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None


# converter (mov -> mp4 / compression)
try:
    from content_extractors.converter import convert_mov_to_mp4
except Exception:
    try:
        from converter import convert_mov_to_mp4
    except Exception:
        convert_mov_to_mp4 = None


def _vtt_to_plain(vtt_text: str) -> str:
    lines = vtt_text.splitlines()
    out = []
    ts_re = re.compile(
        r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")
    for ln in lines:
        s = ln.strip()
        if not s or s.upper().startswith("WEBVTT"):
            continue
        if ts_re.search(ln):
            continue
        if s.isdigit():
            continue
        out.append(s)
    return "\n".join(out)


class ExtractorUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AIVidExtractor — שליפה")
        self.queue: "queue.Queue[str]" = queue.Queue()

        # Menu (Hebrew)
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="פתח כתובת (URL)", command=self._focus_url)
        file_menu.add_separator()
        file_menu.add_command(label="יציאה", command=root.quit)
        menubar.add_cascade(label="קובץ", menu=file_menu)
        root.config(menu=menubar)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        hdr = ttk.Frame(frm)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="URL:", font=(None, 10)).pack(side=tk.LEFT)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(hdr, textvariable=self.url_var, width=70)
        self.url_entry.pack(side=tk.LEFT, padx=8, expand=True, fill=tk.X)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=8)
        self.audio_btn = ttk.Button(
            btns, text="🔊  חילוץ אודיו", command=self.extract_audio)
        self.video_btn = ttk.Button(
            btns, text="🎞️  חילוץ וידאו", command=self.extract_video)
        self.trans_btn = ttk.Button(
            btns, text="📝  חילוץ תמליל", command=self.extract_transcript)
        self.audio_btn.pack(side=tk.LEFT, padx=6)
        self.video_btn.pack(side=tk.LEFT, padx=6)
        self.trans_btn.pack(side=tk.LEFT, padx=6)
        self.compress_btn = ttk.Button(
            btns, text="🗜️  דחוס וידאו", command=self.compress_video)
        self.compress_btn.pack(side=tk.LEFT, padx=6)

        log_frame = ttk.Frame(frm)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(log_frame, height=12,
                           state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scroll.set)

        self.root.after(100, self._poll_queue)

    def _focus_url(self):
        self.url_entry.focus_set()

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append_log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _enqueue(self, msg: str):
        self.queue.put(msg)

    def extract_audio(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("שגיאה", "הזן כתובת URL לפני הפעולה.")
            return
        if get_sound is None:
            messagebox.showerror("לא זמין", "פונקציית חילוץ האודיו לא קיימת.")
            return
        target = filedialog.asksaveasfilename(defaultextension='.mp3', filetypes=[
                                              ('MP3', '*.mp3')], title='שמור אודיו בשם')
        if not target:
            return
        threading.Thread(target=self._audio_worker, args=(
            url, target), daemon=True).start()

    def extract_video(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("שגיאה", "הזן כתובת URL לפני הפעולה.")
            return
        if get_video is None:
            messagebox.showerror("לא זמין", "פונקציית חילוץ הווידאו לא קיימת.")
            return
        target = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[
                                              ('MP4', '*.mp4')], title='שמור וידאו בשם')
        if not target:
            return
        threading.Thread(target=self._video_worker, args=(
            url, target), daemon=True).start()

    def extract_transcript(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("שגיאה", "הזן כתובת URL לפני הפעולה.")
            return
        target = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[
                                              ('Text', '*.txt')], title='שמור תמליל בשם')
        if not target:
            return
        threading.Thread(target=self._transcript_worker,
                         args=(url, target), daemon=True).start()

    def compress_video(self):
        # ask for input mov and output mp4
        input_path = filedialog.askopenfilename(filetypes=[(
            'MOV', '*.mov'), ('MP4', '*.mp4'), ('All', '*.*')], title='בחר קובץ וידאו (MOV)')
        if not input_path:
            return
        output = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[
                                              ('MP4', '*.mp4')], title='שמור וידאו דחוס בשם')
        if not output:
            return
        if convert_mov_to_mp4 is None:
            messagebox.showerror(
                "לא זמין", "פונקציית המרת/דחיסת וידאו לא קיימת.")
            return
        threading.Thread(target=self._compress_worker, args=(
            input_path, output), daemon=True).start()

    # Workers
    def _audio_worker(self, url: str, target: str):
        self._enqueue(f"[אודיו] התחלת הורדה: {url} -> {target}")
        try:
            res = get_sound(url, target)
            if res:
                self._enqueue(f"[אודיו] הושלם: {res}")
            else:
                self._enqueue("[אודיו] נכשל להוריד אודיו")
        except Exception as e:
            self._enqueue(f"[אודיו] שגיאה: {e}")
            self._enqueue(traceback.format_exc())

    def _video_worker(self, url: str, target: str):
        self._enqueue(f"[וידאו] התחלת הורדה: {url} -> {target}")
        try:
            res = get_video(url, target)
            if res:
                self._enqueue(f"[וידאו] הושלם: {res}")
            else:
                self._enqueue("[וידאו] נכשל להוריד וידאו")
        except Exception as e:
            self._enqueue(f"[וידאו] שגיאה: {e}")
            self._enqueue(traceback.format_exc())

    def _transcript_worker(self, url: str, target: str):
        self._enqueue(f"[תמליל] התחלה: {url}")
        parsed = urlparse(url)
        net = (parsed.netloc or "").lower()
        try:
            # prefer youtube
            if ("youtube.com" in net or "youtu.be" in net) and yt_get_transcript:
                self._enqueue("[תמליל] שימוש ב-YouTube extractor")
                text = yt_get_transcript(url)
                if text:
                    with open(target, 'w', encoding='utf-8') as f:
                        f.write(text)
                    self._enqueue(f"[תמליל] נשמר ל: {target}")
                    return
            # tiktok
            if "tiktok.com" in net and tiktok_get_transcript:
                self._enqueue("[תמליל] שימוש ב-TikTok extractor")
                text = tiktok_get_transcript(url)
                if text:
                    with open(target, 'w', encoding='utf-8') as f:
                        f.write(text)
                    self._enqueue(f"[תמליל] נשמר ל: {target}")
                    return

            # fallback: try yt_dlp metadata for subtitles
            if yt_dlp is None:
                self._enqueue("[תמליל] yt_dlp לא מותקן - אין ברירת מחדל")
                return
            self._enqueue("[תמליל] ניסיון ברירת מחדל עם yt-dlp")
            ydl_opts = {'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            caps = info.get('automatic_captions') or info.get(
                'subtitles') or {}
            if not caps:
                self._enqueue('[תמליל] לא נמצאו כתוביות במטא')
                return
            # prefer 'en' then first available
            lang = 'en' if 'en' in caps else next(iter(caps), None)
            entries = caps.get(lang) or []
            subtitle_url = None
            for e in entries:
                if isinstance(e, dict) and e.get('url'):
                    subtitle_url = e['url']
                    break
            if not subtitle_url:
                self._enqueue('[תמליל] לא נמצא URL לכתוביות')
                return
            self._enqueue(f'[תמליל] מוריד כתוביות מ: {subtitle_url}')
            import requests

            r = requests.get(subtitle_url, timeout=30)
            r.raise_for_status()
            text = r.text
            if 'vtt' in r.headers.get('content-type', '').lower() or subtitle_url.endswith('.vtt'):
                text = _vtt_to_plain(text)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(text)
            self._enqueue(f"[תמליל] נשמר ל: {target}")
        except Exception as e:
            self._enqueue(f"[תמליל] שגיאה: {e}")
            self._enqueue(traceback.format_exc())

    def _compress_worker(self, input_path: str, output_path: str):
        self._enqueue(f"[דחיסה] התחלה: {input_path} -> {output_path}")
        try:
            convert_mov_to_mp4(input_path, output_path)
            self._enqueue(f"[דחיסה] הושלם: {output_path}")
        except Exception as e:
            self._enqueue(f"[דחיסה] שגיאה: {e}")
            self._enqueue(traceback.format_exc())


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    ExtractorUI(root)
    root.geometry('900x480')
    root.mainloop()


if __name__ == '__main__':
    main()
