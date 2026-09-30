# tests/test_cjk_words.py

"""M4: words_from_text must count CJK code points individually.

A space-less CJK string has no whitespace, so the old `re.findall(r"\\w+")`
treated the whole string as a single "word". That made n_words scale with the
message count (not content length), keeping CJK conversations below the big
threshold so they were never restored or saved.
"""

import hashing as hs


def test_single_cjk_char_is_one_word():
    assert hs.words_from_text("你") == ["你"]


def test_spaceless_cjk_counts_per_char():
    # A space-less CJK string is one word per character, not one word total.
    assert hs.words_from_text("你好世界") == ["你", "好", "世", "界"]


def test_latin_tokenization_unchanged():
    assert hs.words_from_text("hello world foo") == ["hello", "world", "foo"]


def test_mixed_cjk_and_latin():
    assert hs.words_from_text("hello 世界 ok") == ["hello", "世", "界", "ok"]


def test_cjk_count_scales_with_content_length():
    # Core M4 defect: the count must scale with content length, not message
    # count, so a long CJK conversation can cross the big threshold.
    short = hs.words_from_text("你好")
    long = hs.words_from_text("你好" * 300)
    assert len(short) == 2
    assert len(long) == 600
    assert len(long) > len(short)
