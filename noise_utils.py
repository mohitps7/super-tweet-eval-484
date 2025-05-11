'''
Install package: pip install augly[text]
Package repo: https://github.com/facebookresearch/AugLy/tree/main/augly/text
'''

import random
from typing import Callable, Dict, Optional, Union
from datasets import Dataset
import augly.text as tx

# Optional: enable slang replacement
# SLANG_DICT: Dict[str, str] = {
#     "you": "u", "are": "r", "your": "ur", "people": "ppl", "going": "goin",
#     "great": "gr8", "love": "luv", "what": "wut", "picture": "pic",
#     "really": "rlly", "because": "cuz", "please": "plz", "tonight": "tn",
#     "tomorrow": "tmr"
# }

NoiseFunction = Callable[[str], str]

def build_noisifier(level: str = "medium", seed: Optional[int] = None) -> NoiseFunction:
    if seed is not None:
        random.seed(seed)

    if level == "light":
        augmentations = [
            lambda x: tx.simulate_typos(x, aug_char_p=0.07, aug_word_p=0.07),
            # lambda x: tx.replace_similar_chars(x, aug_char_p=0.05, aug_word_p=0.05),
            lambda x: tx.contractions(x, aug_p=0.15),
        ]
    elif level == "medium":
        augmentations = [
            lambda x: tx.simulate_typos(x, aug_char_p=0.1, aug_word_p=0.15),
            # lambda x: tx.replace_words(x, aug_word_p=0.15, mapping=SLANG_DICT),
            # lambda x: tx.replace_similar_chars(x, aug_char_p=0.1, aug_word_p=0.1),
            # lambda x: tx.split_words(x, aug_word_p=0.05),
            lambda x: tx.merge_words(x, aug_word_p=0.05),
            lambda x: tx.contractions(x, aug_p=0.3),
        ]
    elif level == "heavy":
        augmentations = [
            lambda x: tx.simulate_typos(x, aug_char_p=0.2, aug_word_p=0.3),
            # lambda x: tx.replace_words(x, aug_word_p=0.3, mapping=SLANG_DICT),
            lambda x: tx.replace_similar_chars(x, aug_char_p=0.15, aug_word_p=0.1),
            # lambda x: tx.split_words(x, aug_word_p=0.12),
            lambda x: tx.merge_words(x, aug_word_p=0.15),
            lambda x: tx.contractions(x, aug_p=0.5),
        ]
    else:
        raise ValueError("level must be 'light', 'medium', or 'heavy'")

    # def noisify(text: str) -> str:
    #     random.shuffle(augmentations)
    #     for aug in augmentations:
    #         text = aug(text)
    #     return text
    
    # def noisify(text: str) -> str:
    #     random.shuffle(augmentations)
    #     for aug in augmentations:
    #         try:
    #             out = aug(text)
    #             if isinstance(out, list):
    #                 return " ".join(out)
    #             return str(out)
    #         except Exception as e:
    #             print(f"Augmentation failed: {e}")
    #             return text
    #     return text
    
    # def noisify(text: str) -> str:
    #     random.shuffle(augmentations)
    #     for aug in augmentations:
    #         text = aug(text)
    #         if isinstance(text, list):
    #             text = text[0]  # assume we're only using 1 augmentation per call
    #     return text

    def noisify(text: str) -> str:
        random.shuffle(augmentations)
        for aug in augmentations:
            try:
                out = aug(text)
                if isinstance(out, list):
                    out = out[0]
                text = str(out)
            except Exception as e:
                print(f"Augmentation failed: {e}")
                continue
        return text

    return noisify

def add_noise_to_dataset(dataset: Dataset, text_column: str = "text",
                         level: str = "medium", seed: Optional[int] = None,
                         new_column: Optional[str] = None) -> Dataset:
    noisify = build_noisifier(level, seed)
    col_out = new_column or text_column

    def _augment(example: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        example[col_out] = noisify(example[text_column])
        return example

    return dataset.map(_augment, desc=f"AugLy noise ({level})")
