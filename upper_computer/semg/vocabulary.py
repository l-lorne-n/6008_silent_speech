"""Local cue vocabulary; every recording keeps its own immutable word list."""
import json
from pathlib import Path

DEFAULT_WORDS = ("open", "close", "next", "music")


def validate_words(words):
    if not isinstance(words, (list, tuple)) or not words:
        raise ValueError("请至少选择一个词")
    result = []
    for word in words:
        if (not isinstance(word, str) or not 1 <= len(word) <= 32
                or word != word.strip().casefold()
                or not all(c.isalnum() or c in "_-" for c in word)
                or word == "rest"):
            raise ValueError("词语需为1–32个字母、汉字、数字、短横线或下划线；英文用小写，rest保留给静息")
        if word in result:
            raise ValueError(f"词语重复：{word}")
        result.append(word)
    return result


def load_vocabulary(path):
    path = Path(path)
    if not path.exists():
        return list(DEFAULT_WORDS), list(DEFAULT_WORDS)
    data = json.loads(path.read_text(encoding="utf-8"))
    words = validate_words(data["words"])
    selected = validate_words(data["selected"])
    if not set(selected) <= set(words):
        raise ValueError("选中的词不在词表中")
    return words, selected


def save_vocabulary(path, words, selected):
    words, selected = validate_words(words), validate_words(selected)
    if not set(selected) <= set(words):
        raise ValueError("选中的词不在词表中")
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(words=words, selected=selected), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
