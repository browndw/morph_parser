"""Audit orthographic processes in morph_candidates.json.

The audit compares the surface word form with the concatenation of its
segments and attempts to explain spelling adjustments with productive
English orthographic rules. Anything left unexplained is surfaced for
manual review so that fossilised or historical processes can be removed
from training data.

Rules covered (case-insensitive):
- e_drop: silent "e" deleted before vowel-initial suffixes
- consonant_doubling: doubled consonant before vowel suffixes
- y_to_i / y_to_ie: stem-final "y" replaced by "i" or "ie"
- es_insertion: "e" inserted before plural "-s" after sibilant endings

Outputs are written as JSON to the specified output directory alongside
a concise console summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from difflib import SequenceMatcher

from .audit_utils import CandidateLoader, MorphCandidate

VOWELS = set("aeiou")
SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")
RULE_SAMPLE_LIMIT = 15
UNEXPLAINED_SAMPLE_LIMIT = None
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
VOWELS_FOR_DOUBLING = VOWELS | {"y"}
SION_DELETE_ENDINGS = {
    "de",
    "se",
    "te",
    "ze",
    "ce",
    "ge",
    "re",
    "d",
    "t",
}
SION_REPLACE_ENDINGS = {
    "de",
    "se",
    "te",
    "ce",
    "ge",
    "re",
    "d",
    "t",
}
T_TO_S_REPLACE = {"s", "ss"}
E_TO_I_SUFFIXES = {
    "ferous",
    "cide",
    "ous",
    "meter",
    "vore",
    "gen",
    "vorous",
    "er",
    "linear",
    "lite",
    "sol",
    "form",
    "centric",
    "metry",
    "ment",
    "lagnia",
    "genesis",
    "uria",
    "trend",
    "ery",
    "osis",
    "ary",
    "metrical",
    "o",
    "rostral",
    "cidal",
    "al",
    "able",
    "parous",
    "and",
}
DE_TO_S_SUFFIXES = {"ive", "ible", "ion"}
E_TO_IT_SUFFIXES = {"ive", "iform"}
X_TO_C_SUFFIXES = {
    "ic",
    "ical",
    "oid",
    "oida",
    "idae",
    "ina",
    "inae",
    "oidea",
    "itis",
    "itises",
    "ectomy",
    "ology",
    "logy",
    "otomy",
    "ous",
}
S_TO_D_SUFFIXES = {
    "aceae",
    "aceous",
    "ales",
    "eae",
    "ectomy",
    "ida",
    "idae",
    "iform",
    "inae",
    "ine",
    "oidea",
    "oideae",
    "ophyta",
    "opsida",
    "opsis",
    "osis",
}
S_TO_D_PRECEDING_PAIRS = {
    "as",
    "es",
    "is",
    "ns",
    "os",
    "ys",
}
SIS_TO_T_SUFFIXES = {"ase"}
SIS_TO_S_SUFFIXES = {"ize", "ism"}
SIS_DROP_SUFFIXES = {"tron"}
AXIS_S_DROP_SUFFIXES = {"al"}
SUME_TO_P_SUFFIXES = {"tion"}
OR_TO_R_SUFFIXES = {"ess"}
OUS_TO_OSITY_SUFFIXES = {"ity"}
T_EPENTHESIS_SUFFIXES = {
    "aceae",
    "aurant",
    "azepam",
    "eer",
    "er",
    "ic",
    "ics",
    "id",
    "idae",
    "ino",
    "ism",
    "ist",
    "ite",
    "itis",
    "itude",
    "ive",
    "ize",
    "o",
    "odea",
    "oid",
    "osis",
    "ous",
    "ry",
    "taste",
    "tend",
    "ual",
}

# Rules in this set are considered precise enough to count as strict evidence
# on their own when selecting high-quality training data.
STRICT_HIGH_CONFIDENCE_RULES = {
    "consonant_doubling",
    "y_to_i",
    "y_to_ie",
    "es_insertion",
    "f_to_ves",
    "y_to_ie_plural",
    "n_to_m_assimilation",
    "t_to_s_assimilation",
    "combining_o_elision",
    "sion_assimilation",
    "ish_tion_elision",
}

# These rules are broad and can over-explain bad analyses if accepted alone.
STRICT_REVIEW_ONLY_RULES = {
    "e_drop",
    "ate_elision",
    "vowel_coalescence",
}


@dataclass
class OperationSummary:
    op: str
    joined_span: str
    word_span: str
    joined_range: List[int]
    word_range: List[int]


@dataclass
class ExampleSummary:
    word: str
    segments: List[str]
    joined: str
    delta: int
    rules: List[str]
    operations: List[OperationSummary]


class OrthographyAnalyzer:
    def __init__(self) -> None:
        self.total_examined = 0
        self.rule_counts: Counter[str] = Counter()
        self.rule_samples: Dict[str, List[ExampleSummary]] = defaultdict(list)
        self.unexplained_examples: List[ExampleSummary] = []
        self.strict_unexplained_examples: List[ExampleSummary] = []
        self.delta_distribution: Counter[int] = Counter()

    def analyze(self, candidate: MorphCandidate) -> None:
        joined = candidate.joined_segments
        word = candidate.word
        if joined.lower() == word.lower():
            return

        boundaries = self._segment_boundaries(candidate.segments)
        diff_ops = SequenceMatcher(
            None,
            joined.lower(),
            word.lower(),
            autojunk=False,
        ).get_opcodes()
        rules = self._detect_rules(candidate, diff_ops, boundaries)
        summary = self._build_example(candidate, diff_ops, rules)

        self.total_examined += 1
        self.delta_distribution[candidate.length_delta] += 1

        if rules:
            for rule in rules:
                self.rule_counts[rule] += 1
                if len(self.rule_samples[rule]) < RULE_SAMPLE_LIMIT:
                    self.rule_samples[rule].append(summary)
        else:
            if (
                UNEXPLAINED_SAMPLE_LIMIT is None
                or len(self.unexplained_examples) < UNEXPLAINED_SAMPLE_LIMIT
            ):
                self.unexplained_examples.append(summary)

        if not self._is_strictly_explained(rules):
            if (
                UNEXPLAINED_SAMPLE_LIMIT is None
                or len(self.strict_unexplained_examples)
                < UNEXPLAINED_SAMPLE_LIMIT
            ):
                self.strict_unexplained_examples.append(summary)

    # Detection helpers -------------------------------------------------

    def _detect_rules(
        self,
        candidate: MorphCandidate,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
    ) -> List[str]:
        matched: List[str] = []
        joined = candidate.joined_segments.lower()
        word = candidate.word.lower()
        segs_lower = [seg.lower() for seg in candidate.segments]

        if self._is_e_drop(diff_ops, boundaries, joined):
            matched.append("e_drop")
        if self._is_combining_o_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("combining_o_elision")
        if self._is_ic_drop_before_vowel_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("ic_drop_before_vowel_suffix")
        if self._is_us_truncation_before_vowel_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("us_truncation_before_vowel_suffix")
        if self._is_ine_ide_collapse(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("ine_ide_collapse")
        if self._is_n_to_m_assimilation(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("n_to_m_assimilation")
        if self._is_terminal_a_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("terminal_a_elision")
        if self._is_um_truncation_before_vowel_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("um_truncation_before_vowel_suffix")
        if self._is_terminal_y_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("terminal_y_elision")
        if self._is_er_schwa_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("er_schwa_elision")
        if self._is_duplicate_initial_vowel_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("duplicate_initial_vowel_elision")
        if self._is_duplicate_initial_consonant_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("duplicate_initial_consonant_elision")
        if self._is_terminal_t_elision_before_cy(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("terminal_t_elision_before_cy")
        if self._is_terminal_n_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("terminal_n_elision")
        if self._is_consonant_doubling(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("consonant_doubling")
        if self._is_ly_collapse(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("ly_collapse")
        if self._is_ation_ous_collapse(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
        ):
            matched.append("ation_ous_collapse")
        if self._is_sion_assimilation(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sion_assimilation")
        if self._is_ish_tion_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ish_tion_elision")
        if self._is_e_to_i_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("e_to_i_before_suffix")
        if self._is_ce_to_t_before_i_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ce_to_t_before_i_suffix")
        if self._is_de_to_s_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("de_to_s_before_suffix")
        if self._is_e_to_it_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("e_to_it_before_suffix")
        if self._is_ify_to_ification_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ify_to_ification_elision")
        if self._is_ify_to_fier_collapse(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ify_to_fier_collapse")
        if self._is_sis_to_t_before_ic(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sis_to_t_before_ic")
        if self._is_sis_to_s_before_ist(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sis_to_s_before_ist")
        if self._is_sis_to_t_before_ase(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sis_to_t_before_ase")
        if self._is_sis_to_s_before_ize(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sis_to_s_before_ize")
        if self._is_sis_drop_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sis_drop_before_suffix")
        if self._is_axis_s_drop_before_al(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("axis_s_drop_before_al")
        if self._is_sume_to_p_before_tion(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("sume_to_p_before_tion")
        if self._is_or_to_r_before_ess(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("or_to_r_before_ess")
        if self._is_ous_to_osity(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ous_to_osity")
        if self._is_x_to_c_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("x_to_c_before_suffix")
        if self._is_s_to_d_before_suffix(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("s_to_d_before_suffix")
        if self._is_ten_reduction(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ten_reduction")
        if self._is_vowel_coalescence(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("vowel_coalescence")
        if self._is_t_epenthesis_ic(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("t_epenthesis_ic")
        if self._is_ia_suffix_elision(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("ia_suffix_elision")
        if self._is_t_to_s_assimilation(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("t_to_s_assimilation")
        y_rules = self._y_transform(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        )
        matched.extend(y_rules)
        if self._is_es_insertion(diff_ops, segs_lower, word):
            matched.append("es_insertion")
        if self._is_f_to_ves(diff_ops, boundaries, segs_lower, joined, word):
            matched.append("f_to_ves")
        if self._is_ate_elision(diff_ops, boundaries, segs_lower, joined):
            matched.append("ate_elision")
        if self._is_y_to_ie_plural(
            diff_ops,
            boundaries,
            segs_lower,
            joined,
            word,
        ):
            matched.append("y_to_ie_plural")

        return matched

    def _is_e_drop(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "e":
                continue
            if j2 in boundaries or j1 in boundaries:
                return True
        return False

    def _is_sion_assimilation(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            suffix_idx = boundary_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_segment = segments_lower[suffix_idx]
            if not suffix_segment.startswith("sion"):
                continue
            base_idx = boundary_idx
            if base_idx >= len(segments_lower):
                continue
            base_segment = segments_lower[base_idx]
            suffix_start = w2 if op == "replace" else w1
            if suffix_start + 4 > len(word_lower):
                continue
            if word_lower[suffix_start:suffix_start + 4] != "sion":
                continue
            if op == "delete":
                deleted = joined_lower[j1:j2]
                if (
                    deleted
                    and len(deleted) <= 2
                    and base_segment.endswith(deleted)
                    and deleted in SION_DELETE_ENDINGS
                ):
                    return True
            elif op == "replace":
                from_sub = joined_lower[j1:j2]
                to_sub = word_lower[w1:w2]
                if (
                    to_sub == "s"
                    and base_segment.endswith(from_sub)
                    and from_sub in SION_REPLACE_ENDINGS
                ):
                    return True
        return False

    def _is_ce_to_t_before_i_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        TARGET_SUFFIXES = {
            "ial",
            "ive",
            "ous",
            "ian",
            "ific",
            "ism",
            "ion",
        }
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            from_sub = joined_lower[j1:j2]
            if from_sub not in {"ce", "e"}:
                continue
            if word_lower[w1:w2] != "t":
                continue
            if from_sub == "e":
                if j1 <= 0 or joined_lower[j1 - 1] != "c":
                    continue
                if w1 <= 0 or word_lower[w1 - 1] != "c":
                    continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in TARGET_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("ce"):
                continue
            return True
        return False

    def _is_ish_tion_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "sh":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ish"):
                continue
            if not suffix_seg.startswith("tion"):
                continue
            base_end = boundaries[base_idx]
            if j2 != base_end:
                continue
            if w1 + 4 > len(word_lower):
                continue
            if word_lower[w1:w1 + 4] != "tion":
                continue
            return True
        return False

    def _is_e_to_i_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "e":
                continue
            if word_lower[w1:w2] != "i":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in E_TO_I_SUFFIXES:
                continue
            base_seg = segments_lower[seg_idx]
            seg_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            rel_pos = j1 - seg_start
            if rel_pos < len(base_seg) - 4:
                continue
            return True
        return False

    def _is_de_to_s_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "de":
                continue
            if word_lower[w1:w2] != "s":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in DE_TO_S_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("de"):
                continue
            return True
        return False

    def _is_e_to_it_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "e":
                continue
            if word_lower[w1:w2] != "it":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            if j2 != boundaries[seg_idx]:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg.endswith(("se", "te", "pe")):
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in E_TO_IT_SUFFIXES:
                continue
            return True
        return False

    def _is_ify_to_ification_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "yif":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ify"):
                continue
            if suffix_seg != "ification":
                continue
            if w1 != w2:
                continue
            return True
        return False

    def _is_ify_to_fier_collapse(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "fy":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            if segments_lower[seg_idx] != "ify":
                continue
            if seg_idx == 0:
                continue
            seg_end = boundaries[seg_idx]
            if j1 != seg_end - 2 or j2 != seg_end:
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            if segments_lower[suffix_idx] != "er":
                continue
            base_idx = seg_idx - 1
            base_seg = segments_lower[base_idx]
            replace_found = False
            for rop, rj1, rj2, rw1, rw2 in diff_ops:
                if rop != "replace":
                    continue
                boundary_idx = self._boundary_crossed(boundaries, rj1, rj2)
                if boundary_idx != base_idx:
                    continue
                if word_lower[rw1:rw2] != "f":
                    continue
                replaced = joined_lower[rj1:rj2]
                if not base_seg.endswith(replaced):
                    continue
                if rj2 != boundaries[base_idx]:
                    continue
                replace_found = True
                break
            if not replace_found:
                continue
            return True
        return False

    def _is_sis_to_t_before_ic(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        TARGET_SUFFIX = "ic"
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            replaced = joined_lower[j1:j2]
            to_span = word_lower[w1:w2]
            if not replaced.endswith("sis"):
                continue
            if to_span not in {"t", "st"}:
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            if segments_lower[suffix_idx] != TARGET_SUFFIX:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("sis"):
                continue
            # Ensure the replacement consumes the end of the base segment
            base_end = boundaries[base_idx]
            if j2 != base_end:
                continue
            if to_span == "st":
                # Allow "sis" -> "st" where final "s" remains in the base
                # Check that the word still has the base's preceding "s"
                if w1 == 0 or word_lower[w1 - 1] != "s":
                    continue
            return True
        return False

    def _is_sis_to_s_before_ist(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "is":
                continue
            suffix_idx = self._segment_index_for_position(boundaries, j1)
            if suffix_idx is None:
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg != "ist":
                continue
            if suffix_idx == 0:
                continue
            base_idx = suffix_idx - 1
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("sis"):
                continue
            suffix_start = boundaries[base_idx]
            if j1 != suffix_start or j2 != suffix_start + 2:
                continue
            if w1 == 0 or word_lower[w1 - 1] != "s":
                continue
            return True
        return False

    def _is_sis_to_t_before_ase(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "sis":
                continue
            if word_lower[w1:w2] != "t":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in SIS_TO_T_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("sis"):
                continue
            base_end = boundaries[base_idx]
            if j2 != base_end:
                continue
            return True
        return False

    def _is_sis_to_s_before_ize(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted not in {"si", "is"}:
                continue
            base_idx = self._segment_index_for_position(
                boundaries,
                max(j1 - 1, 0),
            )
            if base_idx is None:
                continue
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in SIS_TO_S_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("sis"):
                continue
            base_end = boundaries[base_idx]
            if not (j1 <= base_end and j2 >= base_end):
                continue
            return True
        return False

    def _is_sis_drop_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "sis":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in SIS_DROP_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg.endswith("sis"):
                continue
            base_end = boundaries[base_idx]
            if j2 != base_end:
                continue
            return True
        return False

    def _is_axis_s_drop_before_al(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "s":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in AXIS_S_DROP_SUFFIXES:
                continue
            base_seg = segments_lower[base_idx]
            if base_seg != "axis":
                continue
            base_end = boundaries[base_idx]
            if j2 != base_end:
                continue
            return True
        return False

    def _is_sume_to_p_before_tion(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "e":
                continue
            if word_lower[w1:w2] != "p":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg.endswith("sume"):
                continue
            base_end = boundaries[seg_idx]
            if j2 != base_end:
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in SUME_TO_P_SUFFIXES:
                continue
            if w2 >= len(word_lower) or word_lower[w2] != "t":
                continue
            return True
        return False

    def _is_or_to_r_before_ess(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "o":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg.endswith("or"):
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in OR_TO_R_SUFFIXES:
                continue
            base_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            expected = base_start + len(base_seg) - 2
            if j1 != expected or j2 != expected + 1:
                continue
            if w1 >= len(word_lower) or word_lower[w1] != "r":
                continue
            return True
        return False

    def _is_ous_to_osity(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "u":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg.endswith("ous"):
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in OUS_TO_OSITY_SUFFIXES:
                continue
            base_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            expected = base_start + len(base_seg) - 2
            if j1 != expected or j2 != expected + 1:
                continue
            if w1 >= len(word_lower) or word_lower[w1] != "s":
                continue
            return True
        return False

    def _is_x_to_c_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "x":
                continue
            if word_lower[w1:w2] != "c":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg.endswith("x"):
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if not suffix_seg:
                continue
            if suffix_seg not in X_TO_C_SUFFIXES and not any(
                suffix_seg.startswith(prefix) for prefix in X_TO_C_SUFFIXES
            ):
                continue
            return True
        return False

    def _is_s_to_d_before_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "s":
                continue
            if word_lower[w1:w2] != "d":
                continue
            if j1 == 0:
                continue
            prev_pair = joined_lower[j1 - 1:j2]
            if prev_pair not in S_TO_D_PRECEDING_PAIRS:
                continue
            suffix = word_lower[w2:]
            if suffix not in S_TO_D_SUFFIXES:
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            if j2 != boundaries[seg_idx]:
                continue
            return True
        return False

    def _is_t_to_s_assimilation(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            suffix_idx = boundary_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_segment = segments_lower[suffix_idx]
            if not suffix_segment or suffix_segment[0] != "i":
                continue
            to_sub = word_lower[w1:w2]
            if to_sub not in T_TO_S_REPLACE:
                continue
            if w2 >= len(word_lower) or word_lower[w2] != "i":
                continue
            from_sub = joined_lower[j1:j2]
            if not from_sub or from_sub[-1] not in {"t", "d"}:
                continue
            base_idx = boundary_idx
            if base_idx >= len(segments_lower):
                continue
            base_segment = segments_lower[base_idx]
            if not base_segment.endswith(from_sub):
                continue
            return True
        return False

    def _is_ten_reduction(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "e":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            seg = segments_lower[seg_idx]
            if not seg.endswith("ten"):
                continue
            rel_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            rel_idx = j1 - rel_start
            if rel_idx != len(seg) - 2:
                continue
            if not (j1 > 0 and joined_lower[j1 - 1] == "t"):
                continue
            if not (j2 < len(joined_lower) and joined_lower[j2] == "n"):
                continue
            next_idx = seg_idx + 1
            if next_idx >= len(segments_lower):
                continue
            next_seg = segments_lower[next_idx]
            if not next_seg.startswith("er"):
                continue
            if w1 < len(word_lower) and word_lower[w1] != "n":
                continue
            return True
        return False

    def _is_vowel_coalescence(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if len(deleted) != 1 or deleted not in VOWELS:
                continue
            boundary_idx = None
            for idx, boundary in enumerate(boundaries[:-1]):
                if j1 == boundary:
                    boundary_idx = idx
                    break
            if boundary_idx is None:
                continue
            prev_idx = boundary_idx
            suffix_idx = boundary_idx + 1
            if prev_idx < 0 or suffix_idx >= len(segments_lower):
                continue
            prev_seg = segments_lower[prev_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not prev_seg.endswith(deleted):
                continue
            if not suffix_seg.startswith(deleted):
                continue
            return True
        return False

    def _is_ia_suffix_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "ai":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if (
                base_idx >= len(segments_lower)
                or suffix_idx >= len(segments_lower)
            ):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ia"):
                continue
            if not suffix_seg.startswith("ic"):
                continue
            return True
        return False

    def _is_t_epenthesis_ic(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "insert":
                continue
            inserted = word_lower[w1:w2]
            if inserted != "t":
                continue
            boundary_idx = None
            for idx, boundary in enumerate(boundaries[:-1]):
                if j1 == boundary == j2:
                    boundary_idx = idx
                    break
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            if not base_seg or base_seg[-1] not in VOWELS:
                continue
            suffix_seg = segments_lower[suffix_idx]
            if suffix_seg not in T_EPENTHESIS_SUFFIXES:
                continue
            return True
        return False

    def _is_ly_collapse(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "el":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("le"):
                continue
            if suffix_seg != "ly":
                continue
            return True
        return False

    def _is_ation_ous_collapse(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted not in {"no", "on"}:
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ion"):
                continue
            if suffix_seg != "ous":
                continue
            return True
        return False

    def _is_consonant_doubling(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "insert":
                continue
            inserted = word_lower[w1:w2]
            if len(inserted) != 1:
                continue
            char = inserted
            if char in VOWELS:
                continue
            lookback = j1 - 1 if j1 > 0 else 0
            seg_idx = self._segment_index_for_position(boundaries, lookback)
            if seg_idx is None:
                continue
            base_seg = segments_lower[seg_idx]
            if not base_seg or base_seg[-1] != char:
                continue
            next_seg = (
                segments_lower[seg_idx + 1]
                if seg_idx + 1 < len(segments_lower)
                else ""
            )
            if next_seg and next_seg[0] in VOWELS_FOR_DOUBLING:
                return True

        for idx, boundary in enumerate(boundaries[:-1]):
            pos = boundary
            if pos <= 0 or pos >= len(word_lower):
                continue
            prev_char = word_lower[pos - 1]
            current_char = word_lower[pos]
            if prev_char != current_char:
                continue
            if idx >= len(segments_lower):
                continue
            base_seg = segments_lower[idx]
            if not base_seg or base_seg[-1] != prev_char:
                continue
            next_seg = (
                segments_lower[idx + 1]
                if idx + 1 < len(segments_lower)
                else ""
            )
            if not next_seg or next_seg[0] not in VOWELS_FOR_DOUBLING:
                continue
            joined_char = joined_lower[pos] if pos < len(joined_lower) else ""
            if joined_char == prev_char:
                continue
            return True
        return False

    def _is_combining_o_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "o":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("o"):
                continue
            if not suffix_seg or suffix_seg[0] not in VOWELS:
                continue
            return True
        return False

    def _is_ic_drop_before_vowel_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted not in {"ic", "ci"}:
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ic"):
                continue
            if not suffix_seg:
                continue
            if suffix_seg[0] not in VOWELS:
                continue
            return True
        return False

    def _is_us_truncation_before_vowel_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "us":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("us"):
                continue
            if not suffix_seg:
                continue
            if suffix_seg[0] not in VOWELS:
                continue
            return True
        return False

    def _is_ine_ide_collapse(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "nei":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("ine"):
                continue
            if not suffix_seg.startswith("ide"):
                continue
            return True
        return False

    def _is_n_to_m_assimilation(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            if joined_lower[j1:j2] != "n":
                continue
            if word_lower[w1:w2] != "m":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("n"):
                continue
            if not suffix_seg:
                continue
            if suffix_seg[0] not in {"b", "p", "m"}:
                continue
            return True
        return False

    def _is_terminal_a_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "a":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("a"):
                continue
            first_char = suffix_seg[0]
            if first_char not in VOWELS and first_char != "y":
                continue
            return True
        return False

    def _is_um_truncation_before_vowel_suffix(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "um":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("um"):
                continue
            if not suffix_seg:
                continue
            if suffix_seg[0] not in VOWELS and suffix_seg[0] != "y":
                continue
            return True
        return False

    def _is_terminal_y_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "y":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("y"):
                continue
            if not suffix_seg:
                continue
            first_char = suffix_seg[0]
            if first_char not in VOWELS and first_char != "y":
                continue
            return True
        return False

    def _is_er_schwa_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted != "e":
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            seg = segments_lower[seg_idx]
            if not seg.endswith("er"):
                continue
            seg_len = len(seg)
            if seg_len < 2:
                continue
            seg_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            rel_pos = j1 - seg_start
            if rel_pos != seg_len - 2:
                continue
            suffix_idx = seg_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            suffix_seg = segments_lower[suffix_idx]
            if not suffix_seg:
                continue
            first_char = suffix_seg[0]
            if first_char not in VOWELS and first_char != "y":
                continue
            return True
        return False

    def _is_duplicate_initial_vowel_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted not in VOWELS:
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            if seg_idx == 0:
                continue
            seg = segments_lower[seg_idx]
            seg_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            rel_pos = j1 - seg_start
            if rel_pos not in {0, 1}:
                continue
            if not seg.startswith(deleted):
                continue
            prev_seg = segments_lower[seg_idx - 1]
            if not prev_seg.endswith(deleted):
                continue
            prev_char = joined_lower[j1 - 1] if j1 > 0 else ""
            if prev_char != deleted:
                continue
            return True
        return False

    def _is_duplicate_initial_consonant_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if not deleted or any(ch in VOWELS for ch in deleted):
                continue
            seg_idx = self._segment_index_for_position(boundaries, j1)
            if seg_idx is None:
                continue
            if seg_idx == 0:
                continue
            seg = segments_lower[seg_idx]
            seg_start = boundaries[seg_idx - 1] if seg_idx > 0 else 0
            rel_pos = j1 - seg_start
            if rel_pos not in {0, 1}:
                continue
            if not seg.startswith(deleted):
                continue
            prev_seg = segments_lower[seg_idx - 1]
            if not prev_seg.endswith(deleted):
                continue
            prev_span = joined_lower[j1 - len(deleted):j1]
            if prev_span != deleted:
                continue
            return True
        return False

    def _is_terminal_t_elision_before_cy(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        TARGET_SUFFIXES = {"cy", "ency"}
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            if joined_lower[j1:j2] != "t":
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("t"):
                continue
            if suffix_seg not in TARGET_SUFFIXES:
                continue
            if not any(base_seg.endswith(end) for end in ("ant", "ent", "it")):
                continue
            return True
        return False

    def _is_terminal_n_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        TARGET_SUFFIX_PREFIXES = ("a", "e", "i", "o", "u", "y")
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if deleted not in {"n", "in"}:
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            base_idx = boundary_idx
            suffix_idx = base_idx + 1
            if suffix_idx >= len(segments_lower):
                continue
            base_seg = segments_lower[base_idx]
            suffix_seg = segments_lower[suffix_idx]
            if not base_seg.endswith("n"):
                continue
            if not suffix_seg:
                continue
            if not suffix_seg.startswith(TARGET_SUFFIX_PREFIXES):
                continue
            if not any(
                base_seg.endswith(end)
                for end in ("in", "kin", "tein")
            ):
                continue
            return True
        return False

    def _y_transform(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> List[str]:
        hits: List[str] = []
        for op, j1, j2, w1, w2 in diff_ops:
            if op not in {"replace", "delete"}:
                continue
            if j2 not in boundaries:
                continue
            seg_idx = self._segment_index_for_position(boundaries, j2 - 1)
            if seg_idx is None:
                continue
            if not segments_lower[seg_idx].endswith("y"):
                continue
            from_chars = joined_lower[j1:j2]
            to_chars = word_lower[w1:w2]
            if op == "replace":
                if from_chars == "y":
                    if to_chars == "i":
                        hits.append("y_to_i")
                    elif (
                        len(to_chars) == 2
                        and to_chars.endswith("i")
                        and j1 > 0
                        and joined_lower[j1 - 1] == to_chars[0]
                    ):
                        hits.append("y_to_i")
                    elif to_chars == "ie":
                        hits.append("y_to_ie")
                elif (
                    from_chars.endswith("y")
                    and to_chars == "i"
                    and seg_idx > 0
                    and segments_lower[seg_idx - 1].endswith(from_chars[:-1])
                ):
                    hits.append("y_to_i")
            elif op == "delete":
                next_seg = (
                    segments_lower[seg_idx + 1]
                    if seg_idx + 1 < len(segments_lower)
                    else ""
                )
                if next_seg.startswith("i"):
                    hits.append("y_to_i")
        return hits

    def _is_es_insertion(
        self,
        diff_ops: Iterable[tuple],
        segments_lower: List[str],
        word_lower: str,
    ) -> bool:
        if not segments_lower or segments_lower[-1] not in {"s", "es"}:
            return False
        base = segments_lower[-2] if len(segments_lower) >= 2 else ""
        if base and not any(base.endswith(end) for end in SIBILANT_ENDINGS):
            return False
        for op, _, _, w1, w2 in diff_ops:
            if op != "insert":
                continue
            inserted = word_lower[w1:w2]
            if inserted == "e" and w2 <= len(word_lower):
                if w2 < len(word_lower) and word_lower[w2] == "s":
                    return True
            if inserted == "es":
                return True
        return False

    def _is_f_to_ves(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        if len(segments_lower) < 2:
            return False
        if segments_lower[-1] not in {"s", "es"}:
            return False
        if not word_lower.endswith("s"):
            return False
        for op, j1, j2, w1, w2 in diff_ops:
            if op != "replace":
                continue
            from_sub = joined_lower[j1:j2]
            to_sub = word_lower[w1:w2]
            if from_sub not in {"f", "fe"}:
                continue
            if to_sub not in {"v", "ve"}:
                continue
            seg_idx = self._segment_index_for_position(boundaries, j2 - 1)
            if seg_idx is None:
                continue
            if not segments_lower[seg_idx].endswith(("f", "fe")):
                continue
            return True
        return False

    def _is_ate_elision(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
    ) -> bool:
        if len(segments_lower) < 2:
            return False
        for op, j1, j2, _, _ in diff_ops:
            if op != "delete":
                continue
            deleted = joined_lower[j1:j2]
            if "t" not in deleted or "e" not in deleted:
                continue
            boundary_idx = self._boundary_crossed(boundaries, j1, j2)
            if boundary_idx is None:
                continue
            seg_idx = boundary_idx
            if seg_idx >= len(segments_lower):
                seg_idx = len(segments_lower) - 1
            if segments_lower[seg_idx].endswith("ate"):
                return True
        return False

    def _boundary_crossed(
        self,
        boundaries: List[int],
        j1: int,
        j2: int,
    ) -> Optional[int]:
        for idx, boundary in enumerate(boundaries[:-1]):
            if j1 < boundary <= j2:
                return idx
        return None

    def _is_y_to_ie_plural(
        self,
        diff_ops: Iterable[tuple],
        boundaries: List[int],
        segments_lower: List[str],
        joined_lower: str,
        word_lower: str,
    ) -> bool:
        if len(segments_lower) < 2:
            return False
        base_idx = len(segments_lower) - 2
        base_segment = segments_lower[base_idx]
        suffix = "".join(segments_lower[base_idx + 1:])
        if suffix and not suffix.startswith("ie"):
            return False
        if not base_segment.endswith("y"):
            # Handle cases where trailing "y" appears as a separate segment
            if (
                base_segment == "y"
                and base_idx > 0
                and segments_lower[base_idx - 1].endswith("y")
            ):
                base_idx -= 1
                base_segment = segments_lower[base_idx]
            else:
                return False
        if len(base_segment) < 2 or base_segment[-2] not in CONSONANTS:
            return False
        for op, j1, j2, w1, w2 in diff_ops:
            if joined_lower[j1:j2] != "y":
                continue
            if j1 == 0 or joined_lower[j1 - 1] not in CONSONANTS:
                continue
            if j2 not in boundaries:
                continue
            seg_idx = self._segment_index_for_position(boundaries, j2 - 1)
            if seg_idx != base_idx:
                continue
            if op == "replace" and word_lower[w1:w1 + 2] == "ie":
                return True
            if op == "delete" and word_lower[w1:w1 + 2] == "ie":
                return True
        return False

    # Utility helpers ---------------------------------------------------

    def _segment_boundaries(self, segments: List[str]) -> List[int]:
        boundaries: List[int] = []
        total = 0
        for seg in segments:
            total += len(seg)
            boundaries.append(total)
        return boundaries

    def _segment_index_for_position(
        self,
        boundaries: List[int],
        position: int,
    ) -> Optional[int]:
        for idx, boundary in enumerate(boundaries):
            if position < boundary:
                return idx
        if boundaries:
            return len(boundaries) - 1
        return None

    def _build_example(
        self,
        candidate: MorphCandidate,
        diff_ops: Iterable[tuple],
        rules: List[str],
    ) -> ExampleSummary:
        operations: List[OperationSummary] = []
        for op, j1, j2, w1, w2 in diff_ops:
            operations.append(
                OperationSummary(
                    op=op,
                    joined_span=candidate.joined_segments[j1:j2],
                    word_span=candidate.word[w1:w2],
                    joined_range=[j1, j2],
                    word_range=[w1, w2],
                )
            )
        return ExampleSummary(
            word=candidate.word,
            segments=candidate.segments,
            joined=candidate.joined_segments,
            delta=candidate.length_delta,
            rules=rules,
            operations=operations,
        )

    def _is_strictly_explained(self, rules: List[str]) -> bool:
        if not rules:
            return False
        if any(rule in STRICT_HIGH_CONFIDENCE_RULES for rule in rules):
            return True
        # If all matched rules are broad heuristics, keep for review.
        if all(rule in STRICT_REVIEW_ONLY_RULES for rule in rules):
            return False
        # Mixed/other rules can still count if there are at least two signals.
        return len(set(rules)) >= 2

    # Reporting ---------------------------------------------------------

    def to_report(self) -> Dict:
        return {
            "total_examined": self.total_examined,
            "delta_distribution": dict(
                sorted(self.delta_distribution.items())
            ),
            "rule_counts": dict(self.rule_counts.most_common()),
            "rule_samples": {
                rule: [self._example_to_dict(ex) for ex in samples]
                for rule, samples in self.rule_samples.items()
            },
            "unexplained_count": len(self.unexplained_examples),
            "unexplained_examples": [
                self._example_to_dict(ex) for ex in self.unexplained_examples
            ],
            "strict_unexplained_count": len(self.strict_unexplained_examples),
            "strict_unexplained_examples": [
                self._example_to_dict(ex)
                for ex in self.strict_unexplained_examples
            ],
        }

    def _example_to_dict(self, example: ExampleSummary) -> Dict:
        return {
            "word": example.word,
            "segments": example.segments,
            "joined": example.joined,
            "delta": example.delta,
            "rules": example.rules,
            "operations": [
                {
                    "op": op.op,
                    "joined_span": op.joined_span,
                    "word_span": op.word_span,
                    "joined_range": op.joined_range,
                    "word_range": op.word_range,
                }
                for op in example.operations
            ],
        }


def run_audit(input_path: Path, output_dir: Path) -> None:
    loader = CandidateLoader(input_path)
    analyzer = OrthographyAnalyzer()

    for idx, candidate in enumerate(loader.iter_candidates()):
        analyzer.analyze(candidate)
        if (idx + 1) % 10000 == 0:
            print(f"  processed {idx + 1:,} candidates")

    report = analyzer.to_report()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "orthography_analysis.json"
    unexplained_path = output_dir / "orthography_unexplained.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(unexplained_path, "w", encoding="utf-8") as fh:
        json.dump(report["unexplained_examples"], fh, indent=2)

    print("\n" + "=" * 70)
    print("ORTHOGRAPHY AUDIT SUMMARY")
    print("=" * 70)
    print(f"Examined entries: {report['total_examined']:,}")
    if report["total_examined"]:
        unexplained_pct = (
            report["unexplained_count"] / report["total_examined"]
        ) * 100
    else:
        unexplained_pct = 0.0
    print(
        "Unexplained cases: "
        f"{report['unexplained_count']:,} "
        f"({unexplained_pct:.1f}% )"
    )
    if report["total_examined"]:
        strict_unexplained_pct = (
            report["strict_unexplained_count"] / report["total_examined"]
        ) * 100
    else:
        strict_unexplained_pct = 0.0
    print(
        "Strict unexplained cases: "
        f"{report['strict_unexplained_count']:,} "
        f"({strict_unexplained_pct:.1f}% )"
    )
    print("Rule hits:")
    for rule, count in report["rule_counts"].items():
        print(f"  {rule:20s} {count:8,}")
    print("Saved report to", report_path)
    print("Unexplained cases written to", unexplained_path)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit orthographic transformations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the candidates JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the resulting JSON report",
    )
    args = parser.parse_args()
    run_audit(args.input, args.output_dir)


if __name__ == "__main__":
    main()
