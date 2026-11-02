"""Learning from the fixes a person makes to a saved transcript (corrections.py)."""
BODY = """Weekly review
Date: 12 Oct 2026, 03:00 PM


[00:00:03] Speaker 1: Haan toh kuber netes ka deployment kal hoga aur free you ko bata dena.
[00:00:11] Speaker 2: the budget is approved but the final numbers are pending.
[00:00:20] Speaker 1: Hum log industry mein use hone vaali chizon pe kaam kar rahe honge.
"""


def _save(tmp_path, corrections, body=BODY):
    f = tmp_path / "Weekly review.txt"
    f.write_text(body, encoding="utf-8-sig")
    corrections.snapshot(f, body)
    return f


def test_a_misheard_name_or_term_is_learned_at_once(tmp_path, clean_data):
    f = _save(tmp_path, clean_data)
    f.write_text(BODY.replace("kuber netes", "Kubernetes").replace("free you", "Priya"), encoding="utf-8")
    learned = dict(clean_data.learn_pending())
    assert learned == {"kuber netes": "Kubernetes", "free you": "Priya"}
    assert "kuber netes=Kubernetes" in clean_data.LEARNED.read_text(encoding="utf-8")


def test_learned_rules_apply_to_later_text(tmp_path, clean_data):
    f = _save(tmp_path, clean_data)
    f.write_text(BODY.replace("kuber netes", "Kubernetes").replace("free you", "Priya"), encoding="utf-8")
    clean_data.learn_pending()
    from glossary import Glossary
    g = Glossary()
    assert g.apply("Haan toh kuber netes ka plan") == "Haan toh Kubernetes ka plan"
    assert g.apply("free you ko bata dena") == "Priya ko bata dena"
    assert g.apply("the budget is fine") == "the budget is fine"


def test_speaker_renames_and_rewritten_lines_are_ignored(tmp_path, clean_data):
    f = _save(tmp_path, clean_data)
    edited = BODY.replace("Speaker 1:", "Anita:").replace(
        "Hum log industry mein use hone vaali chizon pe kaam kar rahe honge.", "A completely different sentence typed instead.")
    f.write_text(edited, encoding="utf-8")
    assert clean_data.learn_pending() == []


def test_a_common_word_needs_the_same_fix_twice(tmp_path, clean_data):
    f = _save(tmp_path, clean_data)
    f.write_text(BODY.replace("the budget is approved", "a budget is approved"), encoding="utf-8")
    assert clean_data.learn_pending() == []                       # once could be a one-off edit
    cur = f.read_text(encoding="utf-8")
    back = cur.replace("a budget is approved", "the budget is approved")
    f.write_text(back, encoding="utf-8")
    clean_data.snapshot(f, back)
    f.write_text(back.replace("the budget is approved", "a budget is approved"), encoding="utf-8")
    assert dict(clean_data.learn_pending()) == {"the": "a"}


def test_an_unchanged_file_learns_nothing(tmp_path, clean_data):
    _save(tmp_path, clean_data)
    assert clean_data.learn_pending() == []
