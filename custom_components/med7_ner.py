import logging
from typing import Any, Dict, List, Text, Type
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import ENTITIES
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.training_data import TrainingData
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR],
    is_trainable=True
)
class MedicalNER(GraphComponent, EntityExtractorMixin):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    def __init__(self, config: Dict[Text, Any]) -> None:
        logger.debug("Initializing MedicalNER component...")
        self.tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        logger.debug("MedicalNER component initialized successfully.")

    def train(self, training_data: TrainingData) -> Resource:
        # If no training is required, return an empty resource.
        return Resource("", {})

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # If no special processing is needed for training, return the data as-is.
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        logger.debug("Processing messages with MedicalNER...")
        for message in messages:
            text = message.get("text")
            if text:
                tokens = self.tokenizer(text, return_tensors="pt")
                outputs = self.model(**tokens)
                predictions = torch.argmax(outputs.logits, dim=2)
                
                entities = []
                for token, prediction in zip(tokens.tokens(), predictions[0].numpy()):
                    label = self.model.config.id2label[prediction]
                    if label == "B-SIGN_SYMPTOM":  # O is typically the label for non-entity tokens
                        entities.append({
                            "start": token.start_char,
                            "end": token.end_char,
                            "value": token.text,
                            "entity": "symptoms"
                        })
                message.set(ENTITIES, message.get(ENTITIES, []) + entities)
            else:
                logger.warning("Received a message with no text.")
        logger.debug("Processing with MedicalNER completed.")
        return messages
