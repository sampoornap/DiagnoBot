from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from symspellpy import SymSpell, Verbosity
import os


import logging
logging.basicConfig(
    filename='actions.log',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER], is_trainable=False
)
class SymSpellChecker(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
        bigram_path = os.path.join(os.path.dirname(__file__), "frequency_bigramdictionary_en_243_342.txt")
        
        if not self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            raise Exception("Dictionary file not found")
        
        if not self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2):
            raise Exception("Bigram dictionary file not found")


    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:
        # Spell checking typically does not require training, so we return the resource as is.
        return Resource("", {})

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            self.process_message(example)
        return training_data

    # def process(self, messages: List[Message]) -> List[Message]:

    #     for message in messages:
    #         corrected_text = self.correct_text(message.get("text"))
    #         message.set("text", corrected_text)
    #     return messages
    
    # def correct_text(self, text: Text) -> Text:
    #     suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
    #     if suggestions:
    #         corrected_text = suggestions[0].term
    #         return corrected_text
    #     return text

    def process(self, messages: List[Message]) -> List[Message]:

        for message in messages:
            self.process_message(message)
        return messages

    def process_message(self, message: Message) -> Message:
        text = message.get("text")
        if text:
            corrected_text = self.correct_text(text)
            message.set("text", corrected_text)
        return message

    def correct_text(self, text: Text) -> Text:
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            corrected_text = suggestions[0].term
            return corrected_text
        return text
