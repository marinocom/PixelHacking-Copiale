import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def extract_vocab(json_paths):
    """
    Extracts unique vocabulary tokens from transcription fields in given JSON files.
    """
    vocab = set()
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data.values():
            tokens = item["transcription"].split(" ")
            vocab.update(tokens)
    return sorted(list(vocab))

def build_tokenizer(vocab_tokens, save_path="./tokenizer", max_seq_len=64):
    """
    Builds and saves a WordLevel tokenizer using the specified vocabulary.
    """
    os.makedirs(save_path, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    vocab_dict = {tok: idx for idx, tok in enumerate(special_tokens)}
    current_idx = len(special_tokens)

    for token in vocab_tokens:
        if token not in vocab_dict:
            vocab_dict[token] = current_idx
            current_idx += 1

    # Save vocab.json
    with open(os.path.join(save_path, "vocab.json"), "w") as f:
        json.dump(vocab_dict, f, indent=2)

    # Create Tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", vocab_dict["[BOS]"]),
            ("[EOS]", vocab_dict["[EOS]"]),
        ]
    )
    tokenizer.enable_truncation(max_length=max_seq_len)
    tokenizer.enable_padding(
        direction="right",
        pad_id=vocab_dict["[PAD]"],
        pad_token="[PAD]"
    )

    # Save tokenizer.json
    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    # Re-load tokenizer using PreTrainedTokenizerFast to preserve post-processor
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,  # Load from file to keep post_processor
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        clean_up_tokenization_spaces=False
    )

    fast_tokenizer.save_pretrained(save_path)
    return fast_tokenizer


def debug_tokenizer(tokenizer, test_texts, max_seq_len):
    """
    Encodes and decodes test strings, showing how unknown tokens are handled.
    """
    for label, text in test_texts.items():
        print(f"\nOriginal ({label}): {text}")

        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
        )

        input_ids = encoded.input_ids[0].tolist()
        decoded_with_specials = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)

        print(f"Encoded IDs: {input_ids}")
        print(f"Decoded with specials: {decoded_with_specials}")
        print(f"Decoded (clean): {decoded_clean}")

        if "[UNK]" in decoded_with_specials:
            print(f"'[UNK]' token used for unknown tokens.")
        else:
            print(f"No '[UNK]' used — all tokens known.")

        if decoded_clean != text:
            print(f"Decoding mismatch in '{label}'")
            print(f"Expected: {text}")
            print(f"Got     : {decoded_clean}")

if __name__ == "__main__":
    TRAIN_LABELS_PATH = ""
    VAL_LABELS_PATH = ""
    TOKENIZER_SAVE_DIR = ""
    GLOBAL_MAX_SEQ_LEN = 64

    json_paths = [TRAIN_LABELS_PATH, VAL_LABELS_PATH]

    vocab_tokens = extract_vocab(json_paths)
    tokenizer = build_tokenizer(vocab_tokens, TOKENIZER_SAVE_DIR, GLOBAL_MAX_SEQ_LEN)

    test_texts = {
        "complex": "VerticalLine Rata de mierda i^^ c^. SquareP u__ b SmallDelta LatinSmallLigatureFi Dagger SmallNHook SleepingSymbol Saturn e^^ CapitalLambda l p^. = CapitalGamma w LatinSmallLigatureFi SquareP u__ h z SmallIota e^^ p n^. r^. CapitalLambda o z Fire NorthEastArrow n u^^ j TopHalfIntegral m y^.. m__ l z e^^ + w",
        "simple": "w z u__ b RockSalt Fire i^^ c^. GANG l p^. ="
    }

    debug_tokenizer(tokenizer, test_texts, GLOBAL_MAX_SEQ_LEN)
