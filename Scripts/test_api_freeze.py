#!/usr/bin/env python3
"""Tests for Scripts/api-freeze.py. Run: python3 -I Scripts/test_api_freeze.py (CI's Public API job runs it first)."""
import importlib.util
import os
import unittest

spec = importlib.util.spec_from_file_location("api_freeze", os.path.join(os.path.dirname(os.path.abspath(__file__)), "api-freeze.py"))
api_freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_freeze)


def enum(name, nonexhaustive, children=()):
    node = {"kind": "TypeDecl", "declKind": "Enum", "name": name, "children": list(children)}
    if nonexhaustive:
        node["declAttributes"] = ["Nonexhaustive"]
    return node


def struct(name, children=()):
    return {"kind": "TypeDecl", "declKind": "Struct", "name": name, "children": list(children)}


ROOT = {"children": [
    struct("ModelReadiness", [enum("State", True), enum("Kind", False)]),
    enum("PreparationError", True),
    enum("PreparationStatus", False),
    struct("Other", [enum("State", False)]),
    struct("Options", [enum("Kind", True)]),
]}


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.known = api_freeze.enums(ROOT)

    def classify(self, line):
        return api_freeze.classify(["/* Fixed-layout Type Changes */", line, ""], self.known)

    def test_a_case_added_to_a_nonexhaustive_enum_is_additive(self):
        self.assertEqual(self.classify("EnumElement PreparationError.timeout has been added as a new enum case"),
                         (["EnumElement PreparationError.timeout has been added as a new enum case"], []))

    def test_a_nested_enum_matches_by_its_full_name(self):
        additive, breaking = self.classify("EnumElement ModelReadiness.State.paused has been added as a new enum case")
        self.assertEqual((len(additive), breaking), (1, []))

    def test_a_nested_enum_matches_by_a_shorter_name_only_when_unambiguous(self):
        # "Kind" names an exhaustive ModelReadiness.Kind and a nonexhaustive Options.Kind: not additive.
        additive, breaking = self.classify("EnumElement Kind.extra has been added as a new enum case")
        self.assertEqual((additive, len(breaking)), ([], 1))
        # "State" names ModelReadiness.State (nonexhaustive) and Other.State (exhaustive): not additive.
        self.assertEqual(len(self.classify("EnumElement State.paused has been added as a new enum case")[1]), 1)
        # "ReadinessOnly" style: a short name that is a suffix of nonexhaustive enums only is additive.
        known = api_freeze.enums({"children": [struct("ModelReadiness", [enum("Phase", True)])]})
        self.assertEqual(api_freeze.classify(["EnumElement Phase.waiting has been added as a new enum case"], known)[1], [])

    def test_a_payload_case_is_matched(self):
        additive, _ = self.classify("EnumElement PreparationError.quota(_:) has been added as a new enum case")
        self.assertEqual(len(additive), 1)

    def test_a_case_added_to_an_exhaustive_enum_breaks(self):
        self.assertEqual(len(self.classify("EnumElement PreparationStatus.queued has been added as a new enum case")[1]), 1)

    def test_an_unknown_enum_breaks(self):
        self.assertEqual(len(self.classify("EnumElement Missing.x has been added as a new enum case")[1]), 1)

    def test_every_other_finding_breaks(self):
        self.assertEqual(len(self.classify("Func ModelPreparation.status(_:) has been removed")[1]), 1)


class ExposureTests(unittest.TestCase):
    FREEZE = {"AetherLanguage": ["*"], "AetherAssets": ["ModelReadiness", "AssetReadLease"],
              "AetherKit": ["ModelLifecyclePolicy", "LanguageModelManifest.Platform"]}

    def references(self, *printed):
        root = {"children": [{"kind": "TypeNominal", "printedName": p} for p in printed]}
        return api_freeze.unfrozen_references(root, self.FREEZE)

    def test_frozen_types_pass(self):
        self.assertEqual(self.references("AetherAssets.ModelReadiness.State.Type", "AetherAssets.AssetReadLease?",
                                         "AetherKit.LanguageModelManifest.Platform", "AetherLanguage.AetherStopReason",
                                         "_Concurrency.AsyncStream<AetherAssets.ModelReadiness.State>"), set())

    def test_unfrozen_types_are_reported(self):
        self.assertEqual(self.references("AetherKit.LanguageModelManifest", "AetherAssets.ExternalFileLease?",
                                         "AetherLanguageRuntime.LanguageSampling"),
                         {"AetherKit.LanguageModelManifest", "AetherAssets.ExternalFileLease",
                          "AetherLanguageRuntime.LanguageSampling"})


if __name__ == "__main__":
    unittest.main()
