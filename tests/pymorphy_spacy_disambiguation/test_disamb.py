import pytest
import spacy

from pymorphy_spacy_disambiguation.disamb import Disambiguator, SimilarityWeighting

b = breakpoint

# TODO: 
#   - more test cases for punctuation etc
#   - test cases for corner cases, like no parsing etc.


MODEL_NAME_UA = "uk_core_news_sm"

@pytest.fixture
def nlp():
    nlp = spacy.load(MODEL_NAME_UA)
    return nlp


@pytest.fixture
def doc(nlp):
    txt = "Жив був король. У нього було царство, де жило сто корів і тридцять кіз, 10 дерев, один камінь і один чарівник. Ще, у нього була дочка. Якось король у неї спитав, чи щаслива вона ('Донечко, ..'). Дочка не відповіла, лише сумно подивилась за вікно, на дерева та козу."
    doc = nlp(txt)
    return doc


def test_feats():
    # Test str -> dict conversion
    feat_str = "VERB Aspect=Imp|Number=Plur|Person=3|Tense=Fut|VerbForm=Fin"
    feat_exp = {
        "Aspect": "Imp",
        "Number": "Plur",
        "Person": "3",
        "Tense": "Fut",
        "VerbForm": "Fin",
        "_POS": "VERB",
    }
    r1 = Disambiguator._morph_str_to_dict(feat_str)
    assert r1 == feat_exp

    feat_str = "NOUN Animacy=Inan|Case=Gen|Number=Plur"
    feat_exp = {"Animacy": "Inan", "Case": "Gen", "Number": "Plur", "_POS": "NOUN"}
    r2 = Disambiguator._morph_str_to_dict(feat_str)
    assert r2 == feat_exp


def test_feats_similarity():
    # test (simple) similarity calculation between dictionaries
    m1 = {
        "Aspect": "Imp",
        "Number": "Plur",
        "Person": "3",
        "Tense": "Fut",
        "VerbForm": "Fin",
        "_POS": "VERB",
    }
    m2 = {
        "Animacy": "Inan",
        "Case": "Gen",
        "Number": "Plur",
        "_POS": "NOUN",
    }
    m3 = {
        "Animacy": "Anim",  # ! changed from m2
        "Case": "Gen",
        "Number": "Plur",
        "_POS": "NOUN",
    }

    assert Disambiguator.calculate_morph_similarity(m1, m1) == 1.0
    assert Disambiguator.calculate_morph_similarity(m2, m3) == 0.75  # 3/4 equal


@pytest.mark.now
def test_disamb(doc):
    # Test correct pymorphy2 morphology disambiguation
    token = doc[12]  # корів - many options, but it's cows, not measles
    d = Disambiguator()
    res = d.get_with_disambiguation(token)
    assert res.normal_form == "корова"
    assert str(res.tag) == "NOUN,anim plur,accs"


def _test_disamb_text(doc):
    # Test correct pymorphy2 morphology disambiguation
    for token in doc:
        d = Disambiguator()
        res = d.get_with_disambiguation(token)
        assert res.normal_form == token.lemma_
        assert str(res.tag.POS) == token.pos_


def test_feats_similarity_weighted():
    # test (simple) similarity calculation between dictionaries

    m1 = {
        "Aspect": "Imp",
        "Number": "Plur",
        "Person": "3",
        "Tense": "Fut",
        "VerbForm": "Fin",
        "_POS": "VERB",
    }
    m2 = {
        "_NORMAL_FORM": "whatever",
        "Animacy": "Inan",
        "Case": "Gen",
        "Number": "Plur",
        "_POS": "NOUN",
    }
    m3 = {
        "_NORMAL_FORM": "not_whatever",  # ! changed
        "Animacy": "Anim",  # ! changed from m2
        "Case": "Gen",
        "Number": "Plur",
        "_POS": "NOUN",
    }
    m3_1 = {
        "_NORMAL_FORM": "not_whatever",  # ! changed
        "Animacy": "Inan",  # ! changed from m2
        "Case": "Not genitive",
        "Number": "Sing",
        "_POS": "VERB",
    }

    ## Both below equal except for score
    m3_11 = {
        "_NORMAL_FORM": "not_whatever",
        "Animacy": "Inan",
        "Case": "Not genitive",
        "Number": "Sing",
        "_POS": "VERB",
        "_SCORE": 0.3,
    }
    m3_12 = {
        "_NORMAL_FORM": "not_whatever",
        "Animacy": "Inan",
        "Case": "Not genitive",
        "Number": "Sing",
        "_POS": "VERB",
        "_SCORE": 0.4,
    }

    w_normal = SimilarityWeighting(normal_form=1000)

    assert Disambiguator.weighted_calculate_morph_similarity(m1, m1) == 1.0

    assert Disambiguator.weighted_calculate_morph_similarity(m2, m3) == 0.6  # 3/5 equal
    assert Disambiguator.weighted_calculate_morph_similarity(m1, m1, w_normal) == 1.0

    # everything differs except normal form, but it has a high weight
    assert Disambiguator.weighted_calculate_morph_similarity(m3, m3_1, w_normal) > 0.99

    # 3/5 equal, but the highlgy weighted name differs
    assert Disambiguator.weighted_calculate_morph_similarity(m2, m3, w_normal) < 0.01

    # One with higher score should win
    #  b()
    assert Disambiguator.weighted_calculate_morph_similarity(
        m3, m3_11
    ) < Disambiguator.weighted_calculate_morph_similarity(m3, m3_12)

    # TODO - do some neat examples with real words/morphologies demostrating this weighting mechanism thing
