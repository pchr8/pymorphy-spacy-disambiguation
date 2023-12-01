import logging

logging.basicConfig()
logger = logging.getLogger(__package__)

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any

import spacy
from spacy.tokens import Token, Doc

import pymorphy2
from pymorphy2.tagset import OpencorporaTag
from pymorphy2.analyzer import Parse

from dataclasses import dataclass, field


from russian_tagsets import converters
from russian_tagsets.ud import Tag14

b = breakpoint

#  MODEL_NAME_UA = "uk_core_news_sm"


@dataclass
class SimilarityWeighting:
    """Weights for each of these when calculating similarity, if present.

    The final score is determined by a sum of 1.0*score for each bit
    belonging to one of the classes below, divided by max possible.

    TODO: readable error if no morphological analysis done on spacy token

    IN THE FUTURE:
        1. this could accept a dictionary with keys coming from
        pymorphy2's GRAM_MAP,
            except that our similarity calculation works AFTER translation,
            that is on dictionaries with Universal Dependencies FEATS-like tags
            that don't match to pymorphy's OpencorporaTag anymore
    """

    # Certainty score assigned by pymorphy2 (for Russian only!).
    # This weighting is a multiplier for that score
    score: float = 1.0

    # Whether the lemma / normal_form is equal
    normal_form: float = 1.0

    # Penalty when one of the grammemes/tags is missing in one of the two dicts
    # 0.0 means "do nothing", 1.0 means "substract 1"
    missing_grammeme: float = 0.0

    # All the other cases: the Opencorpora tag classes (below)
    normal_grammeme: float = 1.0

    #  # OpencorporaTag tag classes
    #  # All come from pymorphy2's GRAM_MAP
    #  # (`list(OpencorporaTag.GRAM_MAP)`)
    #  PARTS_OF_SPEECH: float = 1.0
    #  ANIMACY: float = 1.0
    #  GENDERS: float = 1.0
    #  NUMBERS: float = 1.0
    #  CASES: float = 1.0
    #  ASPECTS: float = 1.0
    #  TRANSITIVITY: float = 1.0
    #  PERSONS: float = 1.0
    #  TENSES: float = 1.0
    #  MOODS: float = 1.0
    #  VOICES: float = 1.0
    #  INVOLVEMENT: float = 1.0


class Disambiguator:
    """
    Class that disambiguates between different pymorphy2 parsings
    using data from spacy. Ukrainian-first, Russian probably supported as well. 

    TODO: clean handling of corner cases like no parsing, no POS etc.
        and explicitly handle token without POS / not in dictionary / in dict with Noe
    TODO: do I want to make it usable in Russian too? How much effort to test?

    TODO: allow setting weights from init

    Pymorphy OCT list of tags: https://github.com/pymorphy2/pymorphy2/blob/master/pymorphy2/tagset.py
    """

    POS_KEY = "_POS"
    NORMAL_FORM_KEY = "_NORMAL_FORM"
    SCORE_KEY = "_SCORE"

    # We use the converter's 'list' of OD tag types
    # https://github.com/kmike/russian-tagsets/blob/master/russian_tagsets/ud.py#L17
    #  UD14_grammemes = Tag14.GRAM_MAP # .items() l.129

    def __init__(
        self, pymorphy_analyzer=None, similarity_weights: SimilarityWeighting = None, **kwargs
    ):  # , spacy_pipeline=None):
        # TODO - exceptions if pymorphy2-ua is not downloaded
        self.pymorphy_analyzer = (
            pymorphy_analyzer
            if pymorphy_analyzer
            else pymorphy2.MorphAnalyzer(lang="uk")
        )
        #  self.nlp = spacy_pipeline if spacy_pipeline else spacy.load(MODEL_NAME_UA)
        self.converter = converters.converter("opencorpora-int", "ud14")
        self.weights = (
            similarity_weights if similarity_weights else SimilarityWeighting(**kwargs)
        )

    @staticmethod
    def get_spacy_morph(token: Token):
        sp_morph = token.morph.to_dict()
        sp_morph[Disambiguator.NORMAL_FORM_KEY] = token.lemma_

        sp_morph[Disambiguator.POS_KEY] = token.pos_
        return sp_morph

    def _pym_morph_to_dict(
        self,
        #  tag: OpencorporaTag,
        parse: Parse,
    ) -> dict[str, str]:
        """Convert Pymorphy2's OpencorporaTag to FEATS-like Dict"""
        tag = parse.tag
        pym_ud_str = self.converter(str(tag))
        pym_morph = self._morph_str_to_dict(pym_ud_str)
        pym_morph[Disambiguator.NORMAL_FORM_KEY] = parse.normal_form
        pym_morph[Disambiguator.SCORE_KEY] = parse.score  # ! absent for UA
        return pym_morph

    @staticmethod
    def _morph_str_to_dict(ud_str: str) -> dict[str, str]:
        """Convert strings like `VERB Aspect=Imp|Number=Plur|Person=3`
            to a dictionary representation.

            Basically mimics spacy (MorphAnalysis) token.morph.to_dict(),
            except that it supports a POS tag at the beginning, adding it
            under the "_POS" key.

        Args:
            ud_str (str): strings like
                `VERB Aspect=Imp|Number=Plur|Person=3|Tense=Fut|VerbForm=Fin`

        Returns:
            {
                "Aspect": "Imp",
                "Number": "Plur",
                "Person": "3",
                "Tense": "Fut",
                "VerbForm": "Fin",
                "_POS": "VERB",
            }
        """
        if " " not in ud_str:
            # TODO does this ever happen?
            #  raise ValueError(f"{ud_str} has no space")
            feats_str = ud_str
            pos = None
        else:
            pos, feats_str = ud_str.split(" ")

        feats = dict()
        for kv in feats_str.split("|"):
            if "=" in kv:
                k, v = kv.split("=")
                feats[k] = v
            else:
                pass
        if pos:
            feats[Disambiguator.POS_KEY] = pos
        return feats

    @staticmethod
    def get_best_morphological_analysis_index(
        sp_morph: dict[str, str],
        pym_morphs: list[dict[str, str]],
        weights: SimilarityWeighting = None,
    ):
        """Return **index** of the best pymorphy2 morphological analysis
        in the list,both provided as dictionaries.

        "Best" means "closest to the spacy interpretation".

        Args:
            sp_morph (dict[str, str]): sp_morph
            pym_morphs (list[dict[str, str]]): pym_morphs
        """
        best_sim = -1
        best_morphology_index = None
        for i, m in enumerate(pym_morphs):
            sim = Disambiguator.weighted_calculate_morph_similarity(
                sp_morph, m, weights
            )
            if sim > best_sim:
                best_sim = sim
                best_morphology_index = i
        logger.debug(
            f"The most similar morphology to {sp_morph} is {pym_morphs[i]} with {i=} {best_sim=}"
        )
        return i

    def select_best_pymorphy_parsing(
        self, token: Token, pym_morphs: list[Parse]
    ) -> Parse:
        """Given a spacy Token and a list of Pymorphy2's Parse parsing results,
        pick the most likely one.

        See https://pymorphy2.readthedocs.io/en/stable/user/guide.html#select-correct
            for pymorphy2 docu on selecting the correct parsing result
        """
        if not pym_morphs:
            return None
        # Spacy morphology as dictionary
        sp_morph = self.get_spacy_morph(token)

        # Pymorphy2 parsing results (+ as dicts)
        pym_morphs = self.pymorphy_analyzer.parse(token.text)
        pym_dicts = [self._pym_morph_to_dict(x) for x in pym_morphs]

        # Pick the best one
        best_morph_i = self.get_best_morphological_analysis_index(
            sp_morph=sp_morph, pym_morphs=pym_dicts, weights=self.weights
        )
        best_morph = pym_morphs[best_morph_i]

        res = best_morph
        #  if len(pym_morphs)>1:
        #  b()
        return res

    def get_with_disambiguation(self, token: Token | Doc) -> Parse:
        """Given a spacy Token or Doc, run pymorphy2 and return the best possible
        pymorphy2 analysis/parsing/morphology ("разбор") by using
        the morphology analysis (analyses) done by spacy.

        In my experience, spacy's morphological analysis is better than
        pymorphy's, as the latter is dictionary-based and doesn't take into
        account the context.

        Use-case for this: you need to do inflection in pymorphy but need to
        choose the correct parsing for that. So you use spacy
        for what it's good at, and then pymorphy for what it's good at.

        See https://pymorphy2.readthedocs.io/en/stable/user/guide.html#select-correct
            for pymorphy2 docu on selecting the correct parsing option.
        """
        if isinstance(token, spacy.tokens.Doc):
            # TODO - allow this
            raise NotImplementedError

        # Spacy morphology as dictionary
        sp_morph = self.get_spacy_morph(token)

        # Pymorphy2 parsing results
        pym_morphs = self.pymorphy_analyzer.parse(token.text)

        best_morph = self.select_best_pymorphy_parsing(
            token=token, pym_morphs=pym_morphs
        )

        res = best_morph
        return res

    @staticmethod
    def calculate_morph_similarity(t1: dict[str, str], t2: dict[str, str]) -> float:
        # TODO penalize keys not found in one of them?
        sim = 0
        keys = set(t1.keys())
        keys.update(t2.keys())

        for k in keys:
            if k in t1 and k in t2:
                if t1[k] == t2[k]:
                    sim += 1
        sim /= len(keys)
        return sim

    @staticmethod
    def weighted_calculate_morph_similarity(
        spacy_dict: dict[str, str],
        pym_dict: dict[str, str],
        weights: SimilarityWeighting = SimilarityWeighting(),
    ) -> float:
        """Calculates a weighted score for the similarity of two dicts.

        The score is basically a sum of each bit multiplied by its weight
        divided by the max possible score.

        Args:
            spacy_dict (dict[str, str]): spacy_dict
            pym_dict (dict[str, str]): pym_dict
            weights (SimilarityWeighting): weights

        Returns:
            float:
        """
        sim = 0
        weights_sum = 0  # maximum possible

        keys_sp = set(spacy_dict.keys())
        keys_pym = set(pym_dict.keys())

        # All keys
        keys = keys_sp.copy()
        keys.update(pym_dict.keys())

        # Keys found only in one of the two dictionaries
        keys_not_shared = keys_sp.symmetric_difference(keys_pym)
        keys_not_shared.discard(Disambiguator.SCORE_KEY)

        # First score the common keys
        for k in keys:
            # Add max possible score to sum
            if k == Disambiguator.SCORE_KEY:
                weights_sum += 1.0 * weights.score
            elif k == Disambiguator.NORMAL_FORM_KEY:
                weights_sum += 1.0 * weights.normal_form
            else:
                weights_sum += 1.0 * weights.normal_grammeme

            # Calculate actual score based on equality
            if k in spacy_dict and k in pym_dict:
                if spacy_dict[k] == pym_dict[k]:
                    if k == Disambiguator.NORMAL_FORM_KEY:
                        # Normal form is special
                        sim += 1 * weights.normal_form
                    else:
                        sim += 1 * weights.normal_grammeme
                else:
                    #  not equal
                    pass

        # Penalize for missing keys if needed
        for k in keys_not_shared:
            sim -= weights.missing_grammeme

        if Disambiguator.SCORE_KEY in pym_dict:
            sim += pym_dict[Disambiguator.SCORE_KEY] * weights.score

        final_score = sim / weights_sum
        return final_score
