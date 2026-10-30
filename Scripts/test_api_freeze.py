#!/usr/bin/env python3
"""Tests for Scripts/api-freeze.py. Run: python3 -I Scripts/test_api_freeze.py (CI's Public API job runs it first)."""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
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
        return api_freeze.unfrozen_references(root, self.FREEZE, "AetherLanguage")

    @staticmethod
    def conformance(mangled):
        return {"kind": "Conformance", "name": mangled, "mangledName": mangled}

    def test_frozen_types_pass(self):
        self.assertEqual(self.references("AetherAssets.ModelReadiness.State.Type", "AetherAssets.AssetReadLease?",
                                         "AetherKit.LanguageModelManifest.Platform", "AetherLanguage.AetherStopReason",
                                         "_Concurrency.AsyncStream<AetherAssets.ModelReadiness.State>"), set())

    def test_unfrozen_types_are_reported(self):
        self.assertEqual(self.references("AetherKit.LanguageModelManifest", "AetherAssets.ExternalFileLease?",
                                         "AetherLanguageRuntime.LanguageSampling"),
                         {"AetherKit.LanguageModelManifest", "AetherAssets.ExternalFileLease",
                          "AetherLanguageRuntime.LanguageSampling"})

    def test_generic_constraints_are_checked(self):
        root = {"children": [{"kind": "Function", "genericSig": "<τ_0_0 where τ_0_0 : AetherKit.TensorScalar>",
                              "sugared_genericSig": "<Scalar where Scalar == AetherAssets.ModelReadiness.State>"}]}
        self.assertEqual(api_freeze.unfrozen_references(root, self.FREEZE, "AetherAssets"), {"AetherKit.TensorScalar"})

    def test_conformances_are_checked_by_their_mangled_name(self):
        root = {"children": [{"kind": "TypeDecl", "name": "ModelReadiness", "conformances": [
            self.conformance("$ss8SendableP"),                      # the standard library
            self.conformance("$s14AetherLanguage0A9GenerableP"),    # frozen, through a word substitution
            self.conformance("$s9AetherKit12TensorScalarP"),        # unfrozen
        ]}]}
        self.assertEqual(api_freeze.unfrozen_references(root, self.FREEZE, "AetherAssets"), {"AetherKit.TensorScalar"})

    def test_an_external_types_conformances_count_only_to_the_modules_own_protocols(self):
        # The dump of an extension of Float lists every conformance the build can see, AetherKit's included.
        node = {"kind": "TypeDecl", "name": "Float", "isExternal": True, "conformances": [
            self.conformance("$s9AetherKit12TensorScalarP"), self.conformance("$s14AetherLanguage0A9GenerableP")]}
        self.assertEqual(api_freeze.unfrozen_references({"children": [node]}, self.FREEZE, "AetherLanguage"), set())
        unfrozen = {"AetherLanguage": ["AetherLanguageModel"]}
        self.assertEqual(api_freeze.unfrozen_references({"children": [node]}, unfrozen, "AetherLanguage"),
                         {"AetherLanguage.AetherGenerable"})

    def test_accessors_are_checked(self):
        # A property in a constrained extension carries the constraint only on its accessors.
        prop = {"kind": "Var", "name": "x", "children": [{"kind": "TypeNominal", "printedName": "Swift.Int"}],
                "accessors": [{"kind": "Accessor", "name": "Get",
                               "genericSig": "<Element where Element : AetherKit.TensorScalar>"}]}
        root = {"children": [{"kind": "TypeDecl", "name": "Array", "isExternal": True, "children": [prop]}]}
        self.assertEqual(api_freeze.unfrozen_references(root, self.FREEZE, "AetherLanguage"), {"AetherKit.TensorScalar"})

    def test_an_unreadable_aether_conformance_fails_closed(self):
        root = {"children": [{"kind": "TypeDecl", "name": "X", "conformances": [self.conformance("$s9AetherKit3FooVyxGP")]}]}
        self.assertEqual(len(api_freeze.unfrozen_references(root, self.FREEZE, "AetherKit")), 1)


class DemangleTests(unittest.TestCase):
    def test_names(self):
        for mangled, name in [
            ("$s12AetherAssets14ModelTransportP", "AetherAssets.ModelTransport"),
            ("$s12AetherAssets14ModelReadinessC8ProgressV5PhaseO", "AetherAssets.ModelReadiness.Progress.Phase"),
            ("$s14AetherLanguage0A9GenerableP", "AetherLanguage.AetherGenerable"),
            ("$s14AetherLanguage0aB5ModelV12AvailabilityO", "AetherLanguage.AetherLanguageModel.Availability"),
            ("$s9AetherKit0A0P", "AetherKit.Aether"),
            ("$s14AetherLanguage06GuidedA5ThingP", "AetherLanguage.GuidedAetherThing"),
            ("$s14AetherLanguage09GenerableA0P", "AetherLanguage.GenerableAether"),
            ("$s14AetherLanguage0aB5ModelV0B9ReadinessP", "AetherLanguage.AetherLanguageModel.LanguageReadiness"),
            ("$s9AetherKit0018Stub_KitP", "?"),  # punycode
            ("$ss5ErrorP", None), ("$sSQ", None), ("$s10Foundation4DataV", None),
            ("$s9AetherKit3FooV3BarVAEP", "?"), ("$s9AetherKit0z3FooP", "?"),
        ]:
            self.assertEqual(api_freeze.demangled_name(mangled), name, mangled)


def declaration(name, *children, internal=False, external=False):
    node = {"kind": "TypeDecl", "name": name, "children": list(children)}
    if internal:
        node["isInternal"] = True
    if external:
        node["isExternal"] = True
    return node


def member(name, internal=False):
    node = {"kind": "Var", "name": name}
    if internal:
        node["isInternal"] = True
    return node


class PruneTests(unittest.TestCase):
    MODULE = "AetherAssets"

    def dump(self, *children):
        return {"ABIRoot": {"kind": "Root", "name": self.MODULE, "tool_arguments": ["-I", "/path/to/.build"],
                            "children": [{"kind": "Import", "name": "Foundation"}, *children]}}

    def prune(self, names, dump):
        """Returns (pruned root, None) or (None, the failure message)."""
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, f) for f in ("freeze.json", "dump.json", "out.json")]
            for path, content in zip(paths, ({"modules": {self.MODULE: names}}, dump)):
                with open(path, "w") as f:
                    json.dump(content, f)
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    api_freeze.prune(paths[0], self.MODULE, paths[1], paths[2])
            except SystemExit:
                return None, stderr.getvalue()
            with open(paths[2]) as f:
                return json.load(f)["ABIRoot"], None

    def test_keeps_named_declarations_without_imports_internals_or_tool_arguments(self):
        root, error = self.prune(["ModelReadiness"], self.dump(
            declaration("ModelReadiness", member("state"), member("store", internal=True)),
            declaration("Unrelated"), declaration("Hidden", internal=True)))
        self.assertIsNone(error)
        self.assertNotIn("tool_arguments", root)
        self.assertEqual([d["name"] for d in root["children"]], ["ModelReadiness"])
        self.assertEqual([m["name"] for m in root["children"][0]["children"]], ["state"])

    def test_outer_dot_inner_keeps_only_inner(self):
        root, error = self.prune(["Outer.Inner"], self.dump(declaration("Outer", declaration("Inner"), declaration("Other"))))
        self.assertIsNone(error)
        self.assertEqual([c["name"] for c in root["children"][0]["children"]], ["Inner"])

    def test_star_keeps_every_public_declaration_and_external_extensions(self):
        root, error = self.prune(["*"], self.dump(declaration("A"), declaration("B", internal=True),
                                                  declaration("String", external=True)))
        self.assertIsNone(error)
        self.assertEqual([d["name"] for d in root["children"]], ["A", "String"])

    def test_an_external_extension_is_dropped_unless_the_module_is_frozen_whole(self):
        root, error = self.prune(["String", "A"], self.dump(declaration("A"), declaration("String", external=True)))
        self.assertIsNone(root)
        self.assertIn("frozen declaration String is missing", error)

    def test_a_missing_frozen_declaration_fails(self):
        self.assertIn("frozen declaration Gone is missing", self.prune(["Gone"], self.dump(declaration("A")))[1])
        self.assertIn("frozen Outer.Gone is missing",
                      self.prune(["Outer.Gone"], self.dump(declaration("Outer", declaration("Inner"))))[1])
        self.assertIn("is missing", self.prune(["Hidden"], self.dump(declaration("Hidden", internal=True)))[1])

    def test_an_empty_dump_fails(self):
        self.assertIn("no frozen declarations", self.prune(["*"], self.dump(declaration("H", internal=True)))[1])

    def test_a_dump_of_another_module_fails(self):
        dump = self.dump(declaration("A"))
        dump["ABIRoot"]["name"] = "AetherKit"
        self.assertIn("is not a dump of AetherAssets", self.prune(["*"], dump)[1])

    def test_a_kept_declaration_exposing_an_unfrozen_type_fails(self):
        exposed = declaration("ModelReadiness", {"kind": "TypeNominal", "printedName": "AetherAssets.Unfrozen"})
        _, error = self.prune(["ModelReadiness"], self.dump(exposed, declaration("Unfrozen")))
        self.assertIn("expose unfrozen types: AetherAssets.Unfrozen", error)


if __name__ == "__main__":
    unittest.main()
