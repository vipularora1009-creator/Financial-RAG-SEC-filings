from sec_financial_rag.chunking.fixed_token import FixedTokenChunker

# " the" is exactly one cl100k_base token, so token counts are easy to reason about.
THE = " the"


def token_counts(chunker, chunks):
    return [len(chunker.encoding.encode(c)) for c in chunks]


def test_windows_are_size_tokens_and_step_by_size_minus_overlap():
    chunker = FixedTokenChunker(size=512, overlap=64)
    chunks = chunker.split(THE * 1000)
    # starts at 0, 448, 896 -> the last window is the 104-token tail
    assert token_counts(chunker, chunks) == [512, 512, 104]


def test_empty_text_gives_no_chunks():
    assert FixedTokenChunker().split("") == []


def test_no_chunk_that_is_only_overlap():
    # Puzzle: exactly 512 tokens must be ONE chunk. A second chunk would be the last
    # 64 tokens again (pure duplicate), wasting a top-k slot (learning record 0014).
    chunker = FixedTokenChunker(size=512, overlap=64)
    assert token_counts(chunker, chunker.split(THE * 512)) == [512]


def test_text_shorter_than_overlap_is_kept():
    # e.g. "Item 4. Mine Safety Disclosures: Not applicable." must not vanish.
    chunker = FixedTokenChunker(size=512, overlap=64)
    assert token_counts(chunker, chunker.split(THE * 10)) == [10]
