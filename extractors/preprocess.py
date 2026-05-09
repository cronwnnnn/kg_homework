"""预处理：章节切分、句子切分、文本清洗。"""

from __future__ import annotations

import re
from dataclasses import dataclass


_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十0-9]+章\s*[^\n]{0,40})$", re.MULTILINE)
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+){1,3}\s+[^\n]{0,60})$", re.MULTILINE)
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])\s*")
_REF_NOISE_RE = re.compile(r"\[\d+(?:[-,，]\d+)*\]")  # [1] [1,2] [1-3]
_SPACE_RE = re.compile(r"[ \t\u3000]+")


@dataclass
class Sentence:
    """携带章节/段落信息的句子。"""

    text: str
    chapter: str
    section: str
    paragraph_id: int
    sentence_id: int

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Paragraph:
    text: str
    chapter: str
    section: str
    paragraph_id: int


class TextPreprocessor:
    """中文论文文本切分器。

    工作流：
        raw -> 清洗(去引用/合并空白) -> 按行扫描 -> 维护当前章节/小节 ->
        段落级输出 -> 句子级输出
    """

    def __init__(self, min_sentence_len: int = 6) -> None:
        self.min_sentence_len = min_sentence_len

    @staticmethod
    def _clean_text(raw: str) -> str:
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        text = _REF_NOISE_RE.sub("", text)
        text = _SPACE_RE.sub("", text)
        return text

    def split_paragraphs(self, raw: str) -> list[Paragraph]:
        text = self._clean_text(raw)
        paragraphs: list[Paragraph] = []
        cur_chapter = ""
        cur_section = ""
        pid = 0
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            ch_match = _CHAPTER_RE.match(line)
            if ch_match:
                cur_chapter = ch_match.group(1).strip()
                cur_section = ""
                continue
            sec_match = _SECTION_RE.match(line)
            if sec_match:
                cur_section = sec_match.group(1).strip()
                continue
            if len(line) < 8:
                continue
            paragraphs.append(
                Paragraph(
                    text=line,
                    chapter=cur_chapter,
                    section=cur_section,
                    paragraph_id=pid,
                )
            )
            pid += 1
        return paragraphs

    def split_sentences(self, paragraphs: list[Paragraph]) -> list[Sentence]:
        sentences: list[Sentence] = []
        sid = 0
        for para in paragraphs:
            chunks = [s.strip() for s in _SENT_SPLIT_RE.split(para.text) if s.strip()]
            for c in chunks:
                if len(c) < self.min_sentence_len:
                    continue
                sentences.append(
                    Sentence(
                        text=c,
                        chapter=para.chapter,
                        section=para.section,
                        paragraph_id=para.paragraph_id,
                        sentence_id=sid,
                    )
                )
                sid += 1
        return sentences

    def process(self, raw: str) -> tuple[list[Paragraph], list[Sentence]]:
        paragraphs = self.split_paragraphs(raw)
        sentences = self.split_sentences(paragraphs)
        return paragraphs, sentences

    @staticmethod
    def load_text(file_path: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "preprocess", b"", 0, 1, f"无法识别 {file_path} 的编码"
        )
