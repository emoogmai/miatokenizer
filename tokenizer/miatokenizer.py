from __future__ import annotations #This is only to support 3.8.1 version of python for list[int]
from abc import ABC, abstractmethod
import os
import logging

import tiktoken
import re
from configs.mia import (
    MIA_SIMPLE_TOKENIZER_ENC_REGEXP,
    MIA_SIMPLE_TOKENIZER_DEC_REGEXP,
    MIA_SIMPLE_TOKENIZER_FILEPATH
)


class Tokenizer(ABC):
    """
    Defines the contract for tokenizers in mia. A tokenizer encode a string in different tokens and 
    returns a list of ids that represent each one of them, also a tokenizer must be able to do the inverse
    operation meaning decode a list of token ids into the string that this represent.
    """
    def encode(self, context: str) -> list[int]:
        """
        Encodes a string in different tokens and returns a list of integer values that represent ids
        of each encoded token from the provided string.

        Parameters:

        context(str): A string representing the context to encode.

        Returns:
        A list of integer values that identify each token encoded throught this tokenizer.
        """
        pass

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of integer values that represent tokens from a previously encoded string into the
        original encoded string.

        Parameters:

        list(int): A list of token ids to be decoded into its original encoded string.

        Returns:
        The string that represent the provided list of token ids.
        """
        pass

class TokenizerFactory(ABC):
    """
    Defines the contract for tokenizer factories. A tokenizer factory is required in order to create
    specific tokenizer implementations based on the name and an optional vocabulary object.
    Sub-classes must implement this method in order to be able to create specific tokenizers based in its
    name.
    """
    @abstractmethod
    def createTokenizer(self, name: str, vocab: any = None) -> Tokenizer:
        """
        Parameters:

        name(str): Tokenizer's name to be created.
        vocab(any): Any object that represent and/or implement a vocabulary, this is optional.

        Returns:
        An instance of a Tokenizer
        """
        pass

class MiaTokenizer(Tokenizer):
    def __init__(self, internal_tokenizer: any):
        self._logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)
        self._internal_tokenizer = internal_tokenizer

    def encode(self, context: str) -> list[int]:
        self._logger.info(f"Encoding provided context {context} using internal tokenizer ...")
        return self._internal_tokenizer.encode(context)

    def decode(self, token_ids: list[int]) -> str:
        self._logger.info("Dencoding provided token ids using internal tokenizer ...")
        return self._internal_tokenizer.decode(token_ids)    


class MiaTokenizerFactory(TokenizerFactory):
    def __init__(self):
        self._logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)

    def _get_tokenizer_type(self, name: str) -> tuple:
        self._logger.info(f"Getting tokinier based on provided full name: {name} ...")

        tokenizer_name = None
        tokenizer_model = None

        if '_' in name:
            tokenizer_name = name[:name.index('_')]
            tokenizer_model = name[name.index('_') + 1:]
            self._logger.info(f"Tokenizer's name is {tokenizer_name} and tokenizer's model is {tokenizer_model} ...")

        return (tokenizer_name, tokenizer_model)

    def createTokenizer(self, name: str, vocab: any = None) -> Tokenizer:
        wrapped_tokenizer = None

        tokenizer_name, tokenizer_model = self._get_tokenizer_type(name)

        if tokenizer_name == 'tiktoken':
            # Tokenizer to wrap is the tiktoken implementation and specific model should come in tokenizer model value
            self._logger.info(f"Internal tokenizer will be tiktoken with following encoding {tokenizer_model} ...")
            wrapped_tokenizer = tiktoken.get_encoding(tokenizer_model)
        elif tokenizer_name == 'mia':
            # Tokenizer to wrap is the mia simple tokenizer implementation - only for test purpose and educational
            self._logger.info("Internal tokenizer will be mia that use a simple encoding implementation ...")
            wrapped_tokenizer = MiaBasicTokenizer()
        else:
            self._logger.error(f"Provided tokenizer name {tokenizer_name} is invalid. It was not identified !!!")
            raise Exception('No supported tokenizer')

        return MiaTokenizer(wrapped_tokenizer)       

class MiaBasicTokenizer(Tokenizer):
    def __init__(self):
        self._logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)

        self._logger.info("Configuring both token to id and id to token maps required during encodeing and decoding operations ...")
        self._str_to_int = self._create_vocabulary()
        self._int_to_str = {i:s for s,i in self._str_to_int.items()} 

    def _create_vocabulary(self) -> dict:
        #Pre-process data, meaning split text into word and special characters and getting a list of all of them
        self._logger.info("Creating a vocabulary based on pre-processing of raw data ...")
        pre_processed_data = self._preprocesses_data()
        all_word_tokens = sorted(list(set(pre_processed_data)))
        all_word_tokens.extend(["<|eot|>", "<|unk|>"]) 

        #Once we have all tokens from raw splitted text creates a dictionary that map each token to an integer
        #value that is its position in enumerated list
        vocab = {token:integer for integer,token in enumerate(all_word_tokens)}

        return vocab

    def _preprocesses_data(self) -> list[str]:
        #Loads only one file from the tokenizer data folder
        self._logger.info(f"Reading raw data for mia simple tokenizer from {MIA_SIMPLE_TOKENIZER_FILEPATH} ...")  
        with open(MIA_SIMPLE_TOKENIZER_FILEPATH, "rt") as f:
            raw_data = f.read()

        #Here we apply a basic tokenization (spliting text in words, special symbols, etc.) over all data
        #from prviously read file from data folder. To split simple tokenizer use a regular expression
        self._logger.info("Splitting raw data read into different tokens ...")
        raw_data_tokens = re.split(MIA_SIMPLE_TOKENIZER_ENC_REGEXP, raw_data) 
        raw_data_tokens = [item.strip() for item in raw_data_tokens if item.strip()]

        self._logger.info("Returning raw data tokens ...")
        return raw_data_tokens

    def encode(self, context: str) -> list[int]:
        self._logger.info(f"Encoding input context: '{context}' into a list of token ids using previously computed vocabulary ...")
        #Before to encode we need to 'tokenize' provided text context applying same regular expression that
        #during pre-processing of raw data to look into the vocabulay table for corresponent ids
        splitted_context = re.split(MIA_SIMPLE_TOKENIZER_ENC_REGEXP, context)
        splitted_context = [item.strip() for item in splitted_context if item.strip()]
        splitted_context = [item if item in self._str_to_int else "<|unk|>" for item in splitted_context]        

        #Map each item in splitted context into the correspondent id from vacabulary !!!
        ids = [self._str_to_int[s] for s in splitted_context]
        
        return ids 

    def decode(self, token_ids: list[int]) -> str:
        self._logger.info(f"Dencoding provided token ids into the original context that represents ...")
        #Concatenate each mapped id into decoded context based on the list of provided token ids.
        decoded_context = " ".join([self._int_to_str[i] for i in token_ids])
        decoded_context = re.sub(MIA_SIMPLE_TOKENIZER_DEC_REGEXP, r'\1', decoded_context)
        
        return decoded_context



