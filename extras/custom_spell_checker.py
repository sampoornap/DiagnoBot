from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from spellchecker import SpellChecker


import logging
logging.basicConfig(
    filename='actions.log',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER], is_trainable=False
)
class SpellCheckerComponent(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self._config = config
        self.spell = SpellChecker()

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

    def process(self, messages: List[Message]) -> List[Message]:

        for message in messages:
            self.process_message(message)
        return messages

    def process_message(self, message: Message) -> Message:
        text = message.get("text")
        if text:
            corrected_text = self.correct_spelling(text)
            message.set("text", corrected_text)
        return message

    def correct_spelling(self, text: str) -> str:
        corrected_text = []
        for word in text.split():
            corrected_word = self.spell.correction(word)
            # Ensure corrected_word is not None
            if corrected_word is None:
                corrected_word = word
            corrected_text.append(corrected_word)
            logger.debug(f"Original word: {word}, Corrected word: {corrected_word}")
        return " ".join(corrected_text)
    
