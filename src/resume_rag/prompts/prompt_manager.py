import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

@dataclass
class PromptBundle:
    system: Optional[str]
    user: str

def _format_template(template: str, kwargs: Dict[str, Any]) -> str:
    if not template:
        return ""
    return template.format(**kwargs)

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._prompts: Dict[str, Dict[str, str]] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        for prompt_file in self.prompts_dir.glob("*.md"):
            name = prompt_file.stem
            with open(prompt_file, encoding="utf-8") as f:
                content = f.read()
            system_match = re.search(
                r"## System Message\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
            )
            template_match = re.search(
                r"## Template\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
            )
            system_text = system_match.group(1).strip() if system_match else ""
            template_text = template_match.group(1).strip() if template_match else ""
            self._prompts[name] = {
                "system": system_text,
                "template": template_text,
            }

    def get_bundle(self, name: str, **kwargs: Any) -> PromptBundle:
        p = self._prompts.get(name, {})
        tmpl = p.get("template", "")
        user = _format_template(tmpl, kwargs) if tmpl else ""
        sys_raw = p.get("system", "")
        system: Optional[str] = None
        if sys_raw:
            try:
                system = _format_template(sys_raw, kwargs)
            except KeyError:
                system = sys_raw
        return PromptBundle(system=system, user=user)

    def get_prompt(self, name: str, **kwargs: Any) -> str:
        """User template only (for callers that concatenate or log)."""
        return self.get_bundle(name, **kwargs).user

    def get_messages(self, name: str, **kwargs: Any) -> List[BaseMessage]:
        b = self.get_bundle(name, **kwargs)
        msgs: List[BaseMessage] = []
        if b.system:
            msgs.append(SystemMessage(content=b.system))
        msgs.append(HumanMessage(content=b.user))
        return msgs