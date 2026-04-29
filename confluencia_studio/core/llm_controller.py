"""LLM Controller -- streaming output via Qt signals.

Provides LLM integration for Confluencia Studio via OpenAI-compatible API.
Supports streaming responses and code execution suggestions.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object


SYSTEM_PROMPT = """You are an AI assistant for Confluencia, a circRNA drug discovery platform.
You help users with:
- Drug efficacy prediction (Confluencia MOE ensemble + CTM pharmacokinetics)
- Epitope/MHC binding prediction (Mamba3Lite sequence encoding)
- circRNA multi-omics analysis and survival analysis
- Joint evaluation combining multiple modalities
- Benchmarking and statistical analysis

Provide concise, actionable advice. When writing code, include complete examples.
Keep responses focused on the user's specific task."""


class StudioLLMController(QObject if PYQT_AVAILABLE else object):
    """LLM controller for Confluencia Studio.

    Provides streaming chat interface with the following features:
    - OpenAI-compatible API support (DeepSeek, OpenAI, local models)
    - Streaming responses via Qt signals
    - Context preservation across conversations
    - Code execution suggestions
    - Command recommendations
    """

    if PYQT_AVAILABLE:
        streaming = pyqtSignal(str)           # Partial response text
        finished = pyqtSignal(str)            # Full response
        error = pyqtSignal(str)               # Error message
        command_suggested = pyqtSignal(str)   # Suggested CLI command

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        system_prompt: Optional[str] = None,
        max_history: int = 50,
        temperature: float = 0.7,
    ):
        if PYQT_AVAILABLE:
            super().__init__()

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_history = max_history
        self.temperature = temperature

        self._messages: List[Dict[str, str]] = []
        self._session_id: Optional[str] = None

        # Load API key from config if available
        self._load_config()

    @property
    def api_key_set(self) -> bool:
        return bool(self.api_key)

    def _load_config(self) -> None:
        """Load configuration from workspace settings."""
        try:
            from confluencia_studio.core.workspace import DEFAULT_WORKSPACE_DIR
            config_path = DEFAULT_WORKSPACE_DIR / "llm_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                self.api_key = config.get("api_key", self.api_key)
                self.base_url = config.get("base_url", self.base_url)
                self.model = config.get("model", self.model)
        except Exception:
            pass

    def _save_config(self) -> None:
        """Save configuration to workspace settings."""
        try:
            from confluencia_studio.core.workspace import DEFAULT_WORKSPACE_DIR
            config_path = DEFAULT_WORKSPACE_DIR / "llm_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump({
                    "base_url": self.base_url,
                    "model": self.model,
                }, f, indent=2)
        except Exception:
            pass

    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self.api_key = api_key
        self._save_config()

    def set_model(self, model: str, base_url: Optional[str] = None) -> None:
        """Set the model and optional base URL."""
        self.model = model
        if base_url:
            self.base_url = base_url
        self._save_config()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def chat(self, message: str, stream: bool = True) -> None:
        """Send a chat message and receive a response.

        If PYQT_AVAILABLE, emits signals for streaming output.
        Otherwise, returns the complete response.
        """
        if not self.api_key:
            error_msg = "API key not set. Configure in Settings."
            if PYQT_AVAILABLE:
                self.error.emit(error_msg)
            return error_msg

        # Add user message to history
        self._messages.append({"role": "user", "content": message})

        # Trim history
        if len(self._messages) > self.max_history * 2:
            self._messages = self._messages[-self.max_history * 2:]

        try:
            full_response = self._call_api(message, stream=stream)
            return full_response
        except Exception as e:
            error_msg = f"API error: {e}"
            if PYQT_AVAILABLE:
                self.error.emit(error_msg)
            return error_msg

    def _call_api(self, message: str, stream: bool = True) -> str:
        """Call the LLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build messages with system prompt
        api_messages = [{"role": "system", "content": self.system_prompt}]
        # Add recent history (last 20 messages)
        recent = self._messages[-40:]
        api_messages.extend(recent)

        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream,
            "temperature": self.temperature,
        }

        # Determine endpoint
        if "deepseek" in self.base_url:
            endpoint = f"{self.base_url}/chat/completions"
        else:
            endpoint = f"{self.base_url}/chat/completions"

        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))

            if stream:
                # Handle streaming response
                full_text = ""
                for line in response:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        content = line[6:]
                        if content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(content)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                full_text += delta
                                if PYQT_AVAILABLE:
                                    self.streaming.emit(delta)
                        except json.JSONDecodeError:
                            continue

                # Add assistant response to history
                self._messages.append({"role": "assistant", "content": full_text})
                if PYQT_AVAILABLE:
                    self.finished.emit(full_text)
                return full_text
            else:
                # Non-streaming
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                self._messages.append({"role": "assistant", "content": content})
                if PYQT_AVAILABLE:
                    self.finished.emit(content)
                return content

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"HTTP {e.code}: {error_body}")

    def suggest_command(self, context: str) -> Optional[str]:
        """Suggest a CLI command based on context."""
        prompt = f"Based on this context: '{context}', suggest the most appropriate Confluencia CLI command. Return only the command without explanation."
        suggestion = self._call_api(prompt, stream=False)
        if suggestion and not suggestion.startswith("Error"):
            if PYQT_AVAILABLE:
                self.command_suggested.emit(suggestion.strip())
            return suggestion.strip()
        return None

    def generate_script(self, task: str) -> str:
        """Generate a Python script for a given task."""
        prompt = f"""Generate a Python script for the following task in the Confluencia platform:

Task: {task}

Return ONLY the complete Python code, starting with imports. Do not include markdown code blocks."""
        return self._call_api(prompt, stream=False)
