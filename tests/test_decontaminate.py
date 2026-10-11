import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt.decontam import NGRAM, NgramIndex, window_hashes, words  # noqa: E402

BENCH_A = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends daily."
BENCH_B = "Which of the following best describes the main function of the mitochondria in a eukaryotic cell during respiration today?"


def idx():
    return NgramIndex.from_texts({"a": [BENCH_A], "b": [BENCH_B]})


def test_words_normalise_case_and_punctuation():
    assert words("Hello, World! It's 42.") == ["hello", "world", "it", "s", "42"]


def test_short_text_has_no_windows():
    assert len(window_hashes("only a few words here")) == 0


def test_planted_overlap_is_detected_and_attributed():
    ix = idx()
    doc = "Some unrelated preface text. " + BENCH_B.upper() + " And an unrelated tail."
    assert ix.names[ix.match(doc)] == "b"
    assert ix.names[ix.match(BENCH_A)] == "a"


def test_near_miss_below_13_words_is_not_flagged():
    ix = idx()
    ws = words(BENCH_A)
    near = " ".join(ws[: NGRAM - 1]) + " zebra " + " ".join(ws[NGRAM - 1:NGRAM - 1 + 6])
    assert ix.match(near) is None
    assert ix.match("A completely different document about gardening and the weather in spring and summer time every year.") is None


def test_empty_index_and_roundtrip(tmp_path):
    assert NgramIndex.from_texts({"x": []}).match(BENCH_A) is None
    ix = idx()
    p = tmp_path / "ng.npz"
    ix.save(p)
    again = NgramIndex.load(p)
    assert len(again) == len(ix) and again.names == ["a", "b"] and again.match(BENCH_A) == 0


def test_number_and_letter_sequences_are_not_reference_ngrams():
    bench = "assert f([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) == 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a b c d e f g h i j k l m n"
    ix = NgramIndex.from_texts({"code": [bench]})
    math_page = "Count: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15. Zeros: 0 0 0 0 0 0 0 0 0 0 0 0 0 0. Letters a b c d e f g h i j k l m n."
    assert ix.match(math_page) is None
    real = NgramIndex.from_texts({"b": [BENCH_A]})
    assert real.match("x " + BENCH_A) == 0            # windows with real words still match
