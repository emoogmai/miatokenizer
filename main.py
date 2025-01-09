import os
import logging
import argparse
from utils import logger
from tokenizer.miatokenizer import MiaTokenizerFactory

if __name__ == '__main__':
    logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, required=False, help='Name of the tokenizer implementation to use for test purposes, formatted as follow <nameimplementation>_<modelfromimplementation>')    
    parser.add_argument('--sample_text', type=str, required=True, help='Sample text to tokenize using the specified tokenizer')
    args = parser.parse_args()

    tokenizer_impl_name = args.tokenizer_name if args.tokenizer_name else "tiktoken_gpt2"

    tokenizer = MiaTokenizerFactory().createTokenizer(tokenizer_impl_name)
    
    logger.info(f"Sample text to encode: '{args.sample_text}'")
    encoded_sample_text = tokenizer.encode(args.sample_text)
    logger.info(f"Encoded sample text as token ids: {encoded_sample_text}")

    logger.info(f"Decoding token ids: {encoded_sample_text}")
    decoded_sample_text = tokenizer.decode(encoded_sample_text)
    logger.info(f"Decoded token ids into original sample text: {decoded_sample_text}")

    try:
        logger.info("Asserting encoding/decoding process to verify if it is correct ...")
        assert decoded_sample_text == args.sample_text
        logger.info("Sample text properly encoded/decoded taking account out of vocabulary words !!!")
    except AssertionError as e:
        logger.info(f"There are out of vocabulary words and were encoded/decoded as '<|unk|>' in the provided sample text: {args.sample_text}, this is not an error. A tokenizer with a BPE algorithm implementation as mia advaced deal with this scenarios")    

    logger.info("Encoding/Decoding is correct !!!")
