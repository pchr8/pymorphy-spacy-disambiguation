# pymorphy-spacy-disambiguation
## TL;DR
**Use spacy morphology to pick the correct pymorphy2 morphology out of a list of options.**

```python3
from pymorphy_spacy_disambiguation.disamb import Disambiguator

d = Disambiguator()

txt = "Жив був король. У нього було царство, де жило сто корів і тридцять кіз."
doc = nlp(txt)
token = doc[12]  # корів - many options, but it's cows, not measles

res = d.get_with_disambiguation(token)

res
>>> Parse(word='корів', tag=OpencorporaTag('NOUN,anim plur,accs'), normal_form='корова', score=1.0, methods_stack=((DictionaryAnalyzer(), 'корів', 2063, 10),))
```

## Motivation
The package [pymorphy2/pymorphy2: Morphological analyzer / inflection engine for Russian and Ukrainian languages](https://github.com/pymorphy2/pymorphy2) does both morphological analysis and inflection of Russian and Ukrainian words.

It does this **per-word** (based on both dictionary and probabilistic methods), without looking at the context. This exacerbates the problem of **disambiguation**. 

Some words might be written identically but:
- Be different words because of different stress (heteronyms): [lang-uk/ukrainian-heteronyms-dictionary: Dictionary of heteronyms in the Ukrainian language](https://github.com/lang-uk/ukrainian-heteronyms-dictionary)
- Just be different words/lemmas that converge under certain inflections (корова/кір -> корів)

For the Russian language, pymorphy2 also gives a probability score for the different options, but this is absent for the Ukrainian language and in any case may not be enough in many cases.

How do you pick the correct morphological analysis?

Pymorphy describes (in Russian) the problem of choosing the correct morphological analysis out of multiple options: [Руководство пользователя — Морфологический анализатор pymorphy2](https://pymorphy2.readthedocs.io/en/stable/user/guide.html#select-correct)

Basically the TL;DR is either you know what you're talking about (e.g. living beings) or you use the sentence context.

Spacy also does morphology based on context, and in my experience it's much more precise in many cases.

**This package uses uses spacy morphology to disambiguate between different pymorphy2 options.**

## Why can't we use spacy directly then?
Because pymorphy2 also does _inflection_, and to do it correctly it has to start with the correct form. 

In the example below, to inflect корів you have to know if it's talking about cows or about measles, without the sentence around it you just don't know. 

> У царя не було _корів_.  
> 
> Царю потрібні **корови**.

Мова йде про **корову** чи хвороба **кір**?

## Howto
### Basic
**In**: a spacy Token with morphological analysis  
**Out**: a pymorphy2 `Parse` object with the most likely candidate for morphological analysis

Take a spacy token, here it's cows: 'корів'
```python
txt = "Жив був король. У нього було царство, де жило сто корів і тридцять кіз."
doc = nlp(txt)
token = doc[12]  # корів - many options, but it's cows, not measles
``` 

Pymorphy2 would give you three different options:

```python
pymorphy_analyzer = pymorphy2.MorphAnalyzer(lang="uk")

# options = pymorphy_analyzer.parse("корів")
options = pymorphy_analyzer.parse(token.text)

# options are:
# 	first is "кір" (measles), the other two "корова" (cow) in two different cases.
>>> [Parse(word='корів', tag=OpencorporaTag('NOUN,inan plur,gent'), normal_form='кір', score=1.0, methods_stack=((DictionaryAnalyzer(), 'корів', 498, 11),)),
 Parse(word='корів', tag=OpencorporaTag('NOUN,anim plur,gent'), normal_form='корова', score=1.0, methods_stack=((DictionaryAnalyzer(), 'корів', 2063, 8),)),
 Parse(word='корів', tag=OpencorporaTag('NOUN,anim plur,accs'), normal_form='корова', score=1.0, methods_stack=((DictionaryAnalyzer(), 'корів', 2063, 10),))]
 ```

Compare with spacy morphology:
```python
# spacy 
token.morph
>>> Animacy=Anim|Case=Gen|Gender=Fem|Number=Plur
```

The disambiguator compares the normal form, POS and morphology of each and picks the one most consistent with the context-based spacy version:

```python3
from pymorphy_spacy_disambiguation.disamb import Disambiguator

d = Disambiguator()
res = d.get_with_disambiguation(token)

res
>>> Parse(word='корів', tag=OpencorporaTag('NOUN,anim plur,accs'), normal_form='корова', score=1.0, methods_stack=((DictionaryAnalyzer(), 'корів', 2063, 10),))
k
assert res.normal_form == "корова"
assert str(res.tag) == "NOUN,anim plur,accs"
```

### With weighting
The most likely morphology analysis is picked based on a similarity score. 

It takes into account the following:
- Normal (canonical) form of the word (spacy's `token.lemma` and pymorphy2 `parse.normal_form`)
	- корів -> корова
- pymorhy2 score if present (only for the Russian language)
- grammemes: the key/value pairs like `Animacy: Anim`,  `Case: Gene`. 
	- It's basically what would be in the Con-ll FEATS dictionary, **BUT INCLUDES PART OF SPEECH**
		- pymorphy2 grammemes: [Обозначения для граммем (русский язык) — Морфологический анализатор pymorphy2](https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html#grammeme-docs)

The grammemes get translated to Universal Dependencies FEATS format using the package [kmike/russian-tagsets](https://github.com/kmike/russian-tagsets). 


Based on your use case you might care about some more than others. For this **weighting** 
was implemented.

Spacy can provide more or less features than pymorphy2, and `missing_grammeme` sets 
the penalty for key/value pairs missing. By default there's no penalty, just no `+1`
that would be added if they were present and equal.

```python
@dataclass
class SimilarityWeighting:
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
```

To use:
```python3

	# Assume those two dictionary representation of word morphology.
	# Here everything differs except the normal form
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

	# Weighting that increases the weight for normal form
    w_normal = SimilarityWeighting(normal_form=1000)

	# This weighting would make the similarity approach 1, by decreasing the 
	# importance of all the other fields.
    assert Disambiguator.weighted_calculate_morph_similarity(m3, m3_1, w_normal) > 0.99
```



## TODO
- process
	- describe translating between different tag sets with [kmike/russian-tagsets](https://github.com/kmike/russian-tagsets)
