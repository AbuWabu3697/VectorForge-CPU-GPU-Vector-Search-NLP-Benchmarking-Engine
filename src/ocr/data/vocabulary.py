from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CHARACTERS = string.ascii_lowercase + string.ascii_uppercase + string.digits + " " + ".,!?;:'\"-()/&+%"


@dataclass(frozen=True)
class CharacterVocabulary:
    """Deterministic character vocabulary with index 0 reserved for CTC blank.

    CTC emits a blank between characters when needed. It is a model symbol, not
    a character that can occur in ground-truth text.
    """

    characters: str = DEFAULT_CHARACTERS
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary characters must be unique.")

    @property
    def blank_index(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.characters) + 1

    @property
    def char_to_index(self) -> dict[str, int]:
        return {character: index + 1 for index, character in enumerate(self.characters)}

    @property
    def index_to_char(self) -> dict[int, str]:
        return {index + 1: character for index, character in enumerate(self.characters)}

    def encode(self, text: str) -> list[int]:
        mapping = self.char_to_index
        unknown = sorted(set(text) - set(mapping))
        if unknown:
            raise ValueError(f"Text contains characters outside the vocabulary: {unknown!r}")
        return [mapping[character] for character in text]

    def decode(self, indices: Iterable[int], *, ignore_blank: bool = True) -> str:
        mapping = self.index_to_char
        output: list[str] = []
        for raw_index in indices:
            index = int(raw_index)
            if ignore_blank and index == self.blank_index:
                continue
            if index not in mapping:
                raise ValueError(f"Unknown vocabulary index: {index}")
            output.append(mapping[index])
        return "".join(output)

    def to_dict(self) -> dict[str, str]:
        return {"blank_token": self.blank_token, "characters": self.characters}

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> CharacterVocabulary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(characters=data["characters"], blank_token=data["blank_token"])
