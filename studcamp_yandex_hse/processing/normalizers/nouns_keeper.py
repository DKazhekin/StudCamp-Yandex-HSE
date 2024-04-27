import pymorphy2

from studcamp_yandex_hse.processing.normalizers.base_normalizer import BaseNormalizer
from studcamp_yandex_hse.processing.utils import languages


class NounsKeeper(BaseNormalizer):
    """
    A class for normalizing text by keeping only nouns.
    """

    def __init__(self, language: str, keep_latn: bool = False) -> None:
        self.morph = pymorphy2.MorphAnalyzer(lang=languages[language])
        self.keep_latn = keep_latn

    def normalize(self, text: str) -> str:
        """
        Normalize text by keeping only nouns.
        :param text: source text
        :return: preprcessed text with only nouns
        """
        nouns = []
        for word in text.split():
            p = self.morph.parse(str(word))[0]
            if p.tag.POS == "NOUN":
                nouns.append(p.normal_form)

            if self.keep_latn and "LATN" in p.tag:
                nouns.append(p.normal_form)

        return " ".join(nouns)
