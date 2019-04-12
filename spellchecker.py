import nltk
from app.utils import ArgSingleton
from constants import *
from collections import namedtuple


def in_any_dictionary(_word):
    return (english_dict.check(_word)
            or hinglish_dict.check(_word)
            or english_bad_words_dict.check(_word)
            or hinglish_bad_words_dict.check(_word)
            or whitelisted_dict.check(_word))


spell_suggestor = namedtuple("Suggestor", ["mode", "correction_dict", "pruning_method"])


class SpellCorrector(object):
    __metaclass__ = ArgSingleton
    MODE_NONE = -1
    MODE_WHITELISTED = 0
    MODE_ENGLISH = 1
    MODE_HINGLISH = 2
    MODE_BIGRAM_TRIGRAM = 3

    MAX_EDIT_DISTANCE_THRESHOLD = 2
    DOUBLE_METAPHONE_MAX_EDIT_DISTANCE_THRESHOLD = 2

    list_of_hinglish_order = ['hinglish', 'english', 'whitelisted', 'bigram', 'trigram']
    list_of_english_order = ['whitelisted', 'english', 'hinglish', 'bigram', 'trigram']

    def __init__(self):
        self.suggestors = {
            'hinglish': spell_suggestor(SpellCorrector.MODE_HINGLISH, hinglish_dict,
                                        self._prune_suggestions_using_editdist_dm),
            'english': spell_suggestor(SpellCorrector.MODE_ENGLISH, english_dict,
                                       self._prune_suggestions_using_editdist_dm),
            'whitelisted': spell_suggestor(SpellCorrector.MODE_WHITELISTED, whitelisted_dict,
                                           self._prune_suggestions_using_editdist_dm),
            'bigram': spell_suggestor(SpellCorrector.MODE_BIGRAM_TRIGRAM, bigrams_dict,
                                      self._prune_suggestions_first_element),
            'trigram': spell_suggestor(SpellCorrector.MODE_BIGRAM_TRIGRAM, trigrams_dict,
                                       self._prune_suggestions_first_element)}

    def _prune_suggestions_using_editdist_dm(self, word, suggested_corrections):
        suggested_corrections = map(lambda suggestion: suggestion.lower(), suggested_corrections)
        suggested_corrections = filter(lambda suggestion: suggestion[0] == word[0], suggested_corrections)

        _suggestions = []
        for suggested_word in suggested_corrections:
            e_distance = nltk.edit_distance(word, suggested_word)
            if e_distance <= SpellCorrector.MAX_EDIT_DISTANCE_THRESHOLD:
                _suggestions.append((suggested_word, e_distance))

        suggested_corrections = _suggestions
        _suggestions = []

        word_dms = doublemetaphone(word)  # doublemetaphones of word
        for suggested_word, e_distance in suggested_corrections:
            suggested_word_dms = doublemetaphone(suggested_word)
            dme_distance = 1000
            for dm in word_dms:
                for sw_dm in suggested_word_dms:
                    dme_distance = min(dme_distance, nltk.edit_distance(dm, sw_dm))
            if dme_distance <= SpellCorrector.DOUBLE_METAPHONE_MAX_EDIT_DISTANCE_THRESHOLD:
                _suggestions.append((suggested_word, (e_distance, dme_distance)))

        suggested_corrections = _suggestions
        _suggestions = []
        for suggested_word, (e_distance, dme_distance) in suggested_corrections:
            freq = english_words.get(suggested_word, len(english_words) + 1)
            _suggestions.append((suggested_word, freq))

        # Reranked on usage
        suggested_corrections = sorted(_suggestions, key=lambda x: x[1])
        suggested_corrections = [suggested_word for suggested_word, _ in suggested_corrections]

        return suggested_corrections

    def _prune_suggestions_first_element(self, word, suggested_corrections):
        if not suggested_corrections:
            return word
        return [suggested_corrections[0]]

    def correct_word(self, word, prefer_hinglish=False):

        word = word.lower().strip()
        mode, corrections = SpellCorrector.MODE_NONE, [word]
        if not word or in_any_dictionary(word):
            return mode, corrections

        if prefer_hinglish:
            suggestor_preference = SpellCorrector.list_of_hinglish_order
        else:
            suggestor_preference = SpellCorrector.list_of_english_order

        for suggestor_key in suggestor_preference:

            suggestor = self.suggestors[suggestor_key]

            pruning_method = suggestor.pruning_method
            enchant_dict = suggestor.correction_dict

            suggestions = pruning_method(word, enchant_dict.suggest(word))

            if suggestions:
                mode, corrections = suggestor.mode, suggestions
                break

        return mode, corrections

    def correct(self, text):

        prefer_hinglish = False
        for word in text.split():
            if hinglish_dict.check(word) or hinglish_bad_words_dict.check(word):
                prefer_hinglish = True
                break

        tokens = []
        for word in text.split():
            if word.startswith('_') and word.endswith('_'):
                tokens.append(word)
            else:
                mode, corrections = self.correct_word(word, prefer_hinglish=prefer_hinglish)
                tokens.append(corrections[0])
        return u" ".join(tokens)
