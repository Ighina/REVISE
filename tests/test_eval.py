from revise.eval import answer_state, answers_equivalent, clean_generation, is_correct, is_insufficient, normalize_answer


def test_normalize():
    assert normalize_answer("The Beatles.") == "beatles"


def test_insufficient_variants():
    assert is_insufficient("INSUFFICIENT")
    assert is_insufficient("Insufficient evidence.")
    assert not is_insufficient("The evidence shows Paris")


def test_correct_modes():
    golds = ["Miquette Giraudy"]
    assert is_correct("Miquette Giraudy", golds, "strict")
    assert not is_correct("The spouse is Miquette Giraudy", golds, "strict")
    assert is_correct("The spouse is Miquette Giraudy", golds, "lenient")
    assert not is_correct("Steve Hillage", golds, "lenient")
    assert answer_state("INSUFFICIENT", golds) == "insufficient"


def test_equivalence_and_cleaning():
    assert answers_equivalent("Paris", "paris.")
    assert not answers_equivalent("Paris", "INSUFFICIENT")
    assert clean_generation("Answer: Paris\nBecause...") == "Paris"
