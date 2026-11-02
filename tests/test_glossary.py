"""The personal glossary: names and terms spelled right, phrase fixes, no rewriting of ordinary words."""
from glossary import Glossary


def test_a_name_gets_its_capital_letter():
    assert Glossary(terms=["Priya"]).apply("priya ne bataya") == "Priya ne bataya"


def test_wrong_equals_right_replaces_a_word_everywhere():
    assert Glossary(terms=["tecknolegy=Technology"]).apply("new tecknolegy plan") == "new Technology plan"


def test_a_phrase_fix_keeps_the_punctuation_around_it():
    g = Glossary(terms=["free you=Priya"])
    assert g.apply("Dena, free you, abhi.") == "Dena, Priya, abhi."


def test_ordinary_words_are_never_snapped_to_a_term():
    g = Glossary(terms=["Priya"])
    assert g.apply("the price is right") == "the price is right"
