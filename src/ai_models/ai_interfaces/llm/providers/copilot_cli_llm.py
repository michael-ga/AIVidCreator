"""
Copilot CLI provider implementing LLMInterface by directly invoking the Copilot CLI.

Execution
---------
- Calls the local binary: `copilot -p "<prompt>"`.
- If a model is provided, adds: `--model <model>`.
- No environment variables are required.
"""

import os
import sys
import subprocess
import re
from typing import Any, Dict, Iterator, List, Optional, Union
import shutil

# Add src directory to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ai_models.ai_interfaces.models import ChatMessage, LLMResponse, LLMStreamChunk
from ai_models.ai_interfaces.llm.protocols import LLMInterface


def _clean_text_by_tokens(text: str) -> str:
    """
    Extract content between <START> and <END> tokens if present.
    If tokens are not found, return the original text.
    """
    start_token = "<START>"
    end_token = "<END>"

    if start_token in text and end_token in text:
        start_idx = text.find(start_token)
        end_idx = text.find(end_token)

        if start_idx < end_idx:
            # Extract content between tokens
            extracted = text[start_idx + len(start_token):end_idx].strip()
            return extracted

    return text


def _messages_to_prompt(messages: Union[str, List[ChatMessage]]) -> str:
    if isinstance(messages, str):
        return messages
    parts: List[str] = []
    for m in messages:
        role = m.role if hasattr(m, "role") else "user"
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts)


def _strip_fences_and_markers(text: str) -> str:
    """Remove common CLI markers and fenced code blocks wrappers."""
    s = text.strip()
    # Remove bullet prefix
    if s.startswith("●"):
        s = s[1:].lstrip()
    # Strip leading success markers/lines like "✓ List directory ..."
    lines = s.splitlines()
    cleaned_lines: List[str] = []
    for ln in lines:
        lns = ln.strip()
        if lns.startswith("✓ ") or lns.startswith("✗ "):
            continue
        cleaned_lines.append(ln)
    s = "\n".join(cleaned_lines).strip()
    # Remove ```json and ``` fences if present
    if s.startswith("```json"):
        s = s[len("```json"):].lstrip()
    if s.endswith("```"):
        s = s[: -3].rstrip()
    return s


def _extract_json_payload(text: str) -> Optional[str]:
    """
    Attempt to extract a JSON object/array from text.
    Strategy:
      1) Strip known CLI markers and fences
      2) If text starts with { or [, try returning the balanced JSON substring
      3) Else, find the first { or [ and return the balanced JSON substring
    """
    s = _strip_fences_and_markers(text)
    # Quick cases
    start_idx = None
    for i, ch in enumerate(s):
        if ch in '{[':
            start_idx = i
            break
    if start_idx is None:
        return None

    # Balanced scan with quote awareness
    stack: List[str] = []
    in_str = False
    esc = False
    end_idx = None
    open_ch = s[start_idx]
    close_ch = '}' if open_ch == '{' else ']'
    stack.append(open_ch)
    for i in range(start_idx + 1, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in '{[':
                stack.append(ch)
                continue
            if ch in '}]':
                if not stack:
                    continue
                top = stack[-1]
                needed = '}' if top == '{' else ']'
                if ch == needed:
                    stack.pop()
                    if not stack:
                        end_idx = i
                        break
                continue
    if end_idx is None:
        return None
    return s[start_idx:end_idx + 1].strip()


def _run_command_with_prompt(prompt: str, model: Optional[str], timeout: Optional[int] = 120) -> str:
    is_windows = os.name == "nt"

    if is_windows:
        # Use a more robust approach: write prompt to temp file and use file input
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name

        try:
            # Build copilot command to read from file
            pwsh_path = shutil.which("pwsh")
            powershell_exe = pwsh_path or shutil.which("powershell") or "powershell"

            ps_cmd = f'copilot --no-color'
            if model:
                ps_cmd += f' --model {model}'
            ps_cmd += f' -p (Get-Content "{temp_file_path}" -Raw)'

            argv: List[str] = [
                powershell_exe,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                ps_cmd,
            ]

            completed = subprocess.run(
                argv,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    else:
        # POSIX: resolve path and avoid shell
        copilot_path = shutil.which("copilot") or "copilot"
        argv: List[str] = [copilot_path, "--no-color"]
        if model:
            argv += ["--model", model]
        argv += ["-p", prompt]
        completed = subprocess.run(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,

        )

    # All output is merged into stdout
    output = completed.stdout or ""
    if completed.returncode != 0:
        hint = ""
        lower = output.lower()
        if "not recognized" in lower or "not found" in lower:
            hint = " Ensure 'copilot' is installed and on your PATH."
        raise RuntimeError(f"Copilot CLI command failed (exit={completed.returncode}): {output.strip()}{hint}")

    return output


class CopilotCliLLM(LLMInterface):
    """
    LLMInterface implementation backed by an external Copilot CLI command.

    Behavior:
    - Invokes `copilot` directly. Ensure it is installed and on PATH.
    - Timeout defaults to 120s; override with COPILOT_CLI_TIMEOUT_SEC env.
    - Model label defaults to "copilot-cli"; override with COPILOT_CLI_MODEL_NAME env.
    """

    def __init__(self) -> None:
        try:
            self.timeout_sec = int(os.getenv("COPILOT_CLI_TIMEOUT_SEC", "120"))
        except ValueError:
            self.timeout_sec = 120
        self.model_name = os.getenv("COPILOT_CLI_MODEL_NAME", "copilot-cli")

    def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = _messages_to_prompt(messages)
        # Prefer explicit model arg; fallback to environment default if provided
        model_to_use = model or os.getenv("COPILOT_CLI_MODEL") or None
        stdout = _run_command_with_prompt(prompt, model_to_use, timeout=self.timeout_sec)

        # Clean text using START/END tokens first
        cleaned = _clean_text_by_tokens(stdout)

        # Trim metrics/footer robustly
        # 1) Prefer removing blocks that start after a blank line with known footer headers
        footer_pattern = re.compile(r"\n\n+(Total usage|Usage by model|Total duration|Total code changes).*", re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(footer_pattern, "", cleaned)
        # 2) Fallback: cut at first triple newline if still verbose
        if "\n\n\n" in cleaned:
            cleaned = cleaned.split("\n\n\n", 1)[0]

        cleaned = cleaned.strip()
        # 3) Remove a leading bullet like '● ' if present
        if cleaned.startswith("●"):
            cleaned = cleaned[1:].lstrip()

        # Build response
        return LLMResponse(
            text=cleaned,
            model=model or self.model_name,
            usage={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            raw_response={"stdout": stdout},
        )

    def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        # Fallback: run once and yield one chunk. Streaming is not supported via this adapter.
        response = self.generate(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        yield LLMStreamChunk(delta=response.text, model=response.model)

    def get_available_models(self) -> List[str]:
        return [self.model_name]

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        return {
            "name": model or self.model_name,
            "provider": "copilot-cli",
            "notes": "Backed by external Copilot CLI. Configure COPILOT_CLI_GENERATE_CMD.",
        }


def get_copilot_cli_client() -> CopilotCliLLM:
    return CopilotCliLLM()


