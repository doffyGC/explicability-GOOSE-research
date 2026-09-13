"""Tests for the run-matrix generator's ERENO configuration.

Checklist ref.: A.3, and the fix for `label_duplication_audit.md`.

`attacks.legitimate` is the flag that decides whether the dataset holds one
stream or two, and getting it wrong is not a visible failure - the generator
succeeds, the CSVs look ordinary, and the defect only shows up in an audit
nobody ran until 2026-09-13. So the two flags this script writes are pinned
here rather than left to a code read.
"""

import unittest

from generate_run_matrix import attacks_properties_text, set_property

ORIGINAL = """# Attack types (true = enabled, false = disabled)
attacks.legitimate=true
attacks.randomReplay=false
attacks.grayhole=false
attacks.orientedGrayhole=false
"""


def value_of(text, key):
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return None


class AttacksPropertiesTests(unittest.TestCase):
    def test_attack_runs_enable_the_grayhole_and_drop_the_legitimate_stream(self):
        text = attacks_properties_text(ORIGINAL, "attack")
        self.assertEqual(value_of(text, "attacks.orientedGrayhole"), "true")
        self.assertEqual(value_of(text, "attacks.legitimate"), "false")

    def test_benign_runs_disable_the_grayhole_and_also_drop_it(self):
        """The benign controls need the same single-stream capture: with both
        streams the impaired copies get a `normal` twin too."""
        text = attacks_properties_text(ORIGINAL, "benign")
        self.assertEqual(value_of(text, "attacks.orientedGrayhole"), "false")
        self.assertEqual(value_of(text, "attacks.legitimate"), "false")

    def test_excluding_the_legitimate_stream_is_the_default(self):
        for family in ("attack", "benign"):
            self.assertEqual(
                value_of(attacks_properties_text(ORIGINAL, family), "attacks.legitimate"),
                "false", family)

    def test_including_it_is_possible_but_has_to_be_asked_for(self):
        text = attacks_properties_text(ORIGINAL, "attack", include_legitimate=True)
        self.assertEqual(value_of(text, "attacks.legitimate"), "true")
        self.assertEqual(value_of(text, "attacks.orientedGrayhole"), "true")

    def test_unrelated_flags_are_left_alone(self):
        text = attacks_properties_text(ORIGINAL, "attack")
        self.assertEqual(value_of(text, "attacks.randomReplay"), "false")
        self.assertEqual(value_of(text, "attacks.grayhole"), "false")
        self.assertIn("# Attack types", text)

    def test_a_file_missing_the_flag_gets_it_appended(self):
        """An older checkout may not carry the key at all; defaulting to
        whatever ERENO's own default happens to be is exactly the failure this
        function exists to prevent."""
        text = attacks_properties_text("attacks.orientedGrayhole=false\n", "attack")
        self.assertEqual(value_of(text, "attacks.legitimate"), "false")

    def test_writing_twice_is_stable(self):
        once = attacks_properties_text(ORIGINAL, "attack")
        twice = attacks_properties_text(once, "attack")
        self.assertEqual(once, twice)


class SetPropertyTests(unittest.TestCase):
    def test_replacing_does_not_duplicate_the_key(self):
        text = set_property(ORIGINAL, "attacks.legitimate", "false")
        self.assertEqual(sum(1 for line in text.splitlines()
                             if line.startswith("attacks.legitimate=")), 1)

    def test_a_value_with_regex_metacharacters_survives(self):
        text = set_property("scenario.path=old\n", "scenario.path", "C:\\\\tmp\\\\a$b")
        self.assertEqual(value_of(text, "scenario.path"), "C:\\\\tmp\\\\a$b")


if __name__ == "__main__":
    unittest.main()
