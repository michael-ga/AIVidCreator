"""
Text parsing and extraction utilities.

Contains all strategies for extracting desired strings from LLM responses,
including JSON extraction, marker-based extraction, and text cleaning.
"""

import json
from typing import Dict, Any, Optional, List


def extract_between_markers(text: str, start_marker: str = "<START>", end_marker: str = "<END>") -> Optional[str]:
    """
    Extract text between start and end markers.

    Args:
        text: Source text to extract from
        start_marker: Start marker (default: "<START>")
        end_marker: End marker (default: "<END>")

    Returns:
        Extracted text between markers, or None if markers not found
    """
    if not text:
        return None

    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None

    start_idx += len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None

    return text[start_idx:end_idx].strip()


def remove_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code block fences from text.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Text with code block fences removed
    """
    if not text:
        return ""

    text = text.strip()

    # Remove ```json
    if text.startswith("```json"):
        text = text[7:].strip()
    # Remove generic ```
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text using multiple strategies.

    Strategies (in order):
    1. Try parsing the entire text as JSON
    2. Extract from START/END markers
    3. Remove markdown blocks and parse
    4. Find balanced JSON object by matching braces

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON dictionary, or None if extraction fails
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from START/END markers
    extracted = extract_between_markers(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Remove markdown and try again
    cleaned = remove_markdown_code_blocks(text)
    if cleaned != text:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find balanced JSON object by matching braces
    start_idx = cleaned.find('{')
    if start_idx != -1:
        try:
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(cleaned)):
                char = cleaned[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = cleaned[start_idx:i + 1]
                        return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def normalize_scenes_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize scenes structure - unwrap nested scenes if present.

    Handles cases where scenes are nested like:
    {"scenes": {"scenes": [...]}} -> {"scenes": [...]}

    Args:
        data: Dictionary that may contain nested scenes structure

    Returns:
        Normalized dictionary with flat scenes array
    """
    if not isinstance(data, dict):
        return data

    scenes_value = data.get('scenes')

    # Check for nested structure: {"scenes": {"scenes": [...]}}
    if isinstance(scenes_value, dict) and isinstance(scenes_value.get('scenes'), list):
        return {"scenes": scenes_value['scenes']}

    # Already correct structure: {"scenes": [...]}
    if isinstance(scenes_value, list):
        return data

    return data


def extract_text_from_response(text: str, use_markers: bool = True, remove_markdown: bool = True) -> str:
    """
    Extract clean text from LLM response.

    Applies multiple cleaning strategies:
    1. Extract from START/END markers (if use_markers=True)
    2. Remove markdown code blocks (if remove_markdown=True)
    3. Strip whitespace

    Args:
        text: Raw text from LLM response
        use_markers: Whether to extract from START/END markers
        remove_markdown: Whether to remove markdown code blocks

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    result = text.strip()

    # Extract from markers if present
    if use_markers:
        extracted = extract_between_markers(result)
        if extracted:
            result = extracted

    # Remove markdown if requested
    if remove_markdown:
        result = remove_markdown_code_blocks(result)

    return result.strip()


def parse_titles_from_text(text: str, max_titles: int = 7) -> List[str]:
    """
    Parse titles from text response.

    Expects titles to be one per line, and filters out empty lines.

    Args:
        text: Text containing titles (one per line)
        max_titles: Maximum number of titles to return

    Returns:
        List of title strings
    """
    if not text:
        return []

    titles = [
        title.strip()
        for title in text.strip().split('\n')
        if title.strip()
    ]

    return titles[:max_titles]


def clean_filename(text: str, max_length: int = 30) -> str:
    """
    Clean text to be safe for use as filename.

    Args:
        text: Text to clean
        max_length: Maximum length of resulting filename

    Returns:
        Cleaned filename-safe string
    """
    if not text:
        return ""

    # Keep only alphanumeric, spaces, hyphens, and underscores
    cleaned = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).strip()
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Limit length
    cleaned = cleaned[:max_length]

    return cleaned

