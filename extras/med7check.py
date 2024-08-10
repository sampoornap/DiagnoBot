import logging
import en_core_med7_trf
from typing import Any, Dict, List, Text, Type
# import spacy
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import ENTITIES
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.training_data import TrainingData
med7 = en_core_med7_trf.load()
logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER],
    is_trainable=False
)
class Med7NER(GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    def __init__(self, config: Dict[Text, Any]) -> None:
        logger.debug("Initializing Med7NER component...")
        
        # self.med7 = en_core_med7_trf.load()

        logger.debug("Med7NER component initialized successfully.")

    def train(self, training_data: TrainingData) -> Resource:
        # Spell checking typically does not require training, so we return the resource as is.
        return Resource("", {})

    # def process(self, messages: List[Message]) -> List[Message]:
    #     logger.debug("processing messages with Med7NER...")
    #     for message in messages:
    #         doc = self.med7(message.get("text"))
    #         entities = []
    #         for ent in doc.ents:
    #             if ent.label_ == "DRUG":
    #                 entities.append({
    #                     "start": ent.start_char,
    #                     "end": ent.end_char,
    #                     "value": ent.text,
    #                     "entity": "medicine_name"
    #                 })
    #         message.set(ENTITIES, message.get(ENTITIES, []) + entities)
    #     logger.debug("processing with Med7NER completed.")
    #     return messages
    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # If no special processing is needed for training, return the data as-is.
        return training_data
    
    def process(self, messages: List[Message]) -> List[Message]:
        logger.debug("Processing messages with Med7NER...")
        for message in messages:
            text = message.get("text")
            if text:
                doc = med7(text)
                entities = []
                for ent in doc.ents:
                    if ent.label_ == "DURATION":
                        entities.append({
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "value": ent.text,
                            "entity": "symptom_duration"
                        })
                message.set(ENTITIES, message.get(ENTITIES, []) + entities)
            else:
                logger.warning("Received a message with no text.")
        logger.debug("Processing with Med7NER completed.")
        return messages

    # @classmethod
    # def load(cls) -> "Med7NER":
    #     return cls(cls.get_default_config())
