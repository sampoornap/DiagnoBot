import csv
import os
import random
import smtplib
import ssl
from email.message import EmailMessage

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction
from rasa_sdk.events import UserUtteranceReverted, ActionExecuted
from rasa_sdk.events import AllSlotsReset

import cohere
import logging
import requests
import re


from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# from actions.langchain_diagnosis import generate_diagnosis

from dotenv import load_dotenv
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key = COHERE_API_KEY)

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# genai.configure(api_key=GEMINI_API_KEY)
from openai import OpenAI
client = OpenAI()


logging.basicConfig(
    filename='actions.log',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(self, dispatcher, tracker, domain):
        # Define the messages to send at the start of the conversation
        dispatcher.utter_message(response="utter_introduction")
        dispatcher.utter_message(response="utter_how_can_i_help")

        return []
    
class ActionProvideMedicationNames(Action):

    def name(self) -> Text:
        return "action_provide_medication_names"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        medical_condition = tracker.get_slot('medical_condition')
        logger.info(f"Medical condition received: {medical_condition}")
        # dispatcher.utter_message(text=f"Here are some medications for {medical_condition}")
                                 
        if not medical_condition:
            dispatcher.utter_message(text="Please provide the medical condition you are inquiring about.")
            return []

        medications = self.get_medications_by_condition(medical_condition)
       
        if medications:
            top_medications = random.sample(medications, min(5, len(medications)))  
            response = self.generate_cohere_llm_response(medical_condition, top_medications)

            
            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(text=f"Sorry, I couldn't find any medication info for {medical_condition}.")
        
        return []

    def get_medications_by_condition(self, condition):
        medications = []
        csv_file_path = 'data.csv'
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['medical_condition'].strip().lower() == condition.strip().lower():
                    medications.append(row['medicine_name'])
        return medications

    
    def generate_cohere_llm_response(self, medical_condition: Text, medications: List[Text]) -> Text:
        if not medications:
            prompt = f"I'm sorry, I couldn't find any medications for {medical_condition}"
        
        else:
            medications_list = ', '.join(medications)
            prompt = (f"I am a medical assistant. When asked about medications for a medical condition, "
                      f"I provide a list of relevant medications.\n\n"
                      f"Medical Condition: {medical_condition}\n"
                      f"Medications: {medications_list}\n\n"
                      f"Provide this information pointwise in a friendly and informative manner.")


            response = co.generate(prompt = prompt)

            return response.generations[0].text.strip()

    def generate_gemini_llm_response(self, medical_condition: Text, medications: List[Text]) -> Text:
        if not medications:
            prompt = f"I'm sorry, I couldn't find any medications for {medical_condition}"
        
        else:
            medications_list = ', '.join(medications)
            prompt = (f"I am a medical assistant. When asked about medications for a medical condition, "
                      f"I provide a list of relevant medications.\n\n"
                      f"Medical Condition: {medical_condition}\n"
                      f"Medications: {medications_list}\n\n"
                      f"Provide this information pointwise in a friendly and informative manner.")

            model = genai.GenerativeModel('gemini-1.5-flash')

            response = model.generate_content(prompt)

            return response.text
        

class ActionHandleFollowUpQuestions(Action):
    def name(self) -> str:
        return "action_handle_follow_up_questions"
    
    

    def run(self, dispatcher, tracker, domain):

        last_n_events = 20  
        recent_conversation = []
        
        for event in tracker.events[-last_n_events:]:
            if event.get("event") == "user":
                recent_conversation.append(f"User: {event.get('text')}")
            elif event.get("event") == "bot":
                recent_conversation.append(f"Bot: {event.get('text')}")
        
        user_message = tracker.latest_message.get("text")
        diagnosis_result = tracker.get_slot("current_diagnosis")  
        conversation_memory = recent_conversation  

        follow_up_data = {
            "patient_diagnosis": diagnosis_result,
            "patient_query": user_message,
            "conversation_history": conversation_memory
        }

        response = requests.post("http://localhost:5001/follow_up", json=follow_up_data)
        follow_up_result = response.json()[0]
       
        dispatcher.utter_message(text=follow_up_result)
        suggested_follow_up_question = response.json()[1]
        suggested_follow_up_message = "Suggested Follow Up: " + suggested_follow_up_question
        dispatcher.utter_message(text=suggested_follow_up_message)

        return [FollowupAction("utter_ask_if_need_help")]

class ActionProvideSideEffects(Action):

    def name(self) -> Text:
        return "action_provide_side_effects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        medicine_name = tracker.get_slot('medicine_name')
        logger.info(f"Medicine name received: {medicine_name}")

        # dispatcher.utter_message(text=f"here are the side effects of {medicine_name}")
        if not medicine_name:
            dispatcher.utter_message(text="Please provide the name of the medication you are inquiring about.")
            return []

        side_effects = self.get_side_effects_by_medicine(medicine_name)
        if side_effects:
            response = self.generate_llm_response(medicine_name, side_effects)
            dispatcher.utter_message(text=response)
           
        else:
            dispatcher.utter_message(text=f"Sorry, I couldn't find any side effects info for {medicine_name}.")
        
        return []

    def get_side_effects_by_medicine(self, medicine_name):
        side_effects = None
        csv_file_path = 'data.csv'
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['medicine_name'].strip().lower() == medicine_name.strip().lower():
                    side_effects = row['side_effects']
                    break
        return side_effects 

    def generate_llm_response(self, medicine_name: Text, side_effects: Text) -> Text:
        if not side_effects:
            prompt = f"I'm sorry, I couldn't find any side effects information for {medicine_name}."
        else:
            prompt = (f"I am a medical assistant. When asked about the side effects of a medication, "
                      f"I provide a detailed list of potential side effects.\n\n"
                      f"Medication: {medicine_name}\n"
                      f"Side Effects: {side_effects}\n\n"
                      f"Provide this information pointwise in a friendly and informative manner.")
        
        response = co.generate(prompt = prompt)
        return response.generations[0].text.strip()
    

class ActionClearAllSlots(Action):

    def name(self) -> Text:
        return "action_clear_all_slots"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet(slot, None) for slot in tracker.slots]


class ActionSubmitAndProvideDiagnosis(Action):
    def name(self) -> Text:
        return "action_submit_and_provide_diagnosis"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Making an instant diagnosis...")

        return [FollowupAction("action_submit_and_provide_diagnosis_helper")]   

class ActionSubmitAndProvideDiagnosisHelper(Action):

    def name(self) -> Text:
        return "action_submit_and_provide_diagnosis_helper"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        patient_details = tracker.get_slot("patient_responses")


        # symptoms = tracker.get_slot('symptom_list')
        # symptom_duration = tracker.get_slot('symptom_duration')
        # symptom_severity = tracker.get_slot('symptom_severity')
        contact_info = tracker.get_slot('contact_info')
        # past_health_conditions = tracker.get_slot('past_health_conditions')
        
        # if not all([symptoms, symptom_duration, symptom_severity, contact_info, past_health_conditions]):
        #     return [ActionAskNextQuestion()]
        
        llm_report = patient_details.replace(contact_info, "<contact>")

        doctor_report = self.generate_doctor_report(llm_report) + f"Contact: {contact_info}"
        
        response = self.provide_langchain_response(llm_report)
        dispatcher.utter_message(text=llm_report)
        dispatcher.utter_message(text=response)
        
       
        self.send_email("somename@gmail.com", "Patient Diagnosis Report", doctor_report)
        
        dispatcher.utter_message(text="Sent a report to Dr. XYZ. You should here back in 3 days.")

        # AllSlotsReset()
        
        
        # dispatcher.utter_message(text="All slots have been cleared.")
        
        
        return [SlotSet(slot, None) for slot in tracker.slots] + [FollowupAction("utter_ask_if_need_help")] + [SlotSet("current_diagnosis", response)]
    
    def provide_langchain_response(self, report) -> Text:
        if not report:
            prompt = f"I'm sorry, I couldn't find any information necessary for diagnosis."
            return prompt
        else:
            response = requests.post("http://localhost:5001/diagnosis", json={"text": report})
            langchain_result = response.json()

            return langchain_result


    def generate_cohere_llm_response(self, report) -> Text:
        if not report:
            prompt = f"I'm sorry, I couldn't find any information necessary for diagnosis."
        else:
            prompt = (f"You will respond as a medical doctor with extensive knowledge of rare medical conditions."
                      "Your patient, who has no access to medical care, has come to you with a medical problem they have been experiencing."
                       "Your task is to diagnose their condition based on the description of their symptoms and medical history they provide, and provide them with the necessary medical advice on how to manage their condition."
                        "Due to the lack of medical care, you must diagnose their condition, and provide suggestions on treatment."
                        "Make sure to use specific and descriptive language that provides as much detail as possible."
                        "Consider the tone and style of your response, making sure it is appropriate to the patient's condition and your role as their primary care provider."
                        "Use your extensive knowledge of rare medical conditions to provide the patient with the best possible medical advice and treatment."
                      f"Report: {report}\n"
                      f"Provide the diagnosis along with suggested medications. This response will go to a doctor for verification before being sent to the end user. ")
        
        response = co.generate(prompt = prompt)
        return response.generations[0].text.strip()
    
    def generate_doctor_report(self, report) ->Text:
        if not report:
            prompt = f"I'm sorry, I couldn't find any information necessary for making a report."
        else:
            prompt = (f"The following are responses from a patient. Please compile them into a structured doctor report in the following format:"
                      "Doctor Report:"
                      "Patient Age: [Age]"
                      "Sex: [Sex]"
                      "Symptoms: [Symptoms]"
                      "Duration: [Duration]"
                      "Severity: [Severity]"
                      "Past Health Conditions: [Past Health Conditions]"
                      "Other Details: [Other Details]"
                      "\n\n\n"
                      f"The Patient Responses are: {report}")
            
            response = co.generate(prompt = prompt)
            return response.generations[0].text.strip()
    
    
    def generate_gpt_llm_response(self, report) -> Text:
        if not report:
            prompt = f"I'm sorry, I couldn't find any information necessary for diagnosis."
            return prompt
        else:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "I am a medical assistant. When provided the details of a patient, I provide a general diagnosis. However, I am not a doctor and, therefore, let the patient know that.\n\n"},
                    {"role": "user", "content": f"Report: {report}\n Provide the diagnosis along with suggested medications. This response will go to a doctor for verification before being sent to the end user."}
                ]
                )
            
        return completion.choices[0].message

    def send_email(self, recipient_email: str, subject: str, body: str):
        smtp_server = "smtp.gmail.com"

        port = 465
        senderemail = "dxyz54266@gmail.com"
        receiveremail = recipient_email
        password = os.getenv("PASSWORD")

        em=EmailMessage()
        em['From'] = senderemail
        em['to'] = receiveremail
        em['Subject'] = subject

        em.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(senderemail, password)
            server.sendmail(senderemail, receiveremail, em.as_string())

# class ActionAskIfNeedHelp(Action):

#     def name(self) -> Text:
#         return "action_ask_if_need_help"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         dispatcher.utter_message(text="Do you need help with anything else?")
#         return []

class ActionProvideExplanation(Action):

    def name(self) -> str:
        return "action_provide_explanation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        last_question = tracker.get_slot("last_question")

        if last_question == "symptoms":
            dispatcher.utter_message(response="utter_reason_for_asking_symptoms")
        elif last_question == "symptom_duration":
            dispatcher.utter_message(response="utter_reason_for_asking_duration")
        elif last_question == "past_conditions":
            dispatcher.utter_message(response="utter_reason_for_asking_past")
        elif last_question == "contact_info":
            dispatcher.utter_message(response="utter_reason_for_asking_contact")
        elif last_question == "symptom_severity":
            dispatcher.utter_message(response="utter_reason_for_asking_severity")
        else:
            dispatcher.utter_message(text="I'm not sure why this information is needed.")

        return []



class ActionAskSymptoms(Action):

    def name(self) -> Text:
        return "action_ask_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_symptoms")

        return [SlotSet("last_question", "symptoms")] 

class ActionAskSymptomDuration(Action):

    def name(self) -> Text:
        return "action_ask_symptom_duration"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_symptom_duration")
        return [SlotSet("last_question", "symptom_duration")] 
class ActionAskSymptomSeverity(Action):

    def name(self) -> Text:
        return "action_ask_symptom_severity"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_symptom_severity")
        return [SlotSet("last_question", "symptom_severity")] 
    
class ActionAskPatientAge(Action):

    def name(self) -> Text:
        return "action_ask_patient_age"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_patient_age")
        return [SlotSet("last_question", "patient_age")] 
    
class ActionAskPatientSex(Action):

    def name(self) -> Text:
        return "action_ask_patient_sex"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_patient_sex")
        return [SlotSet("last_question", "patient_sex")] 

class ActionAskContactInfo(Action):

    def name(self) -> Text:
        return "action_ask_contact_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_contact_info")
        return [SlotSet("last_question", "contact_info")]

class ActionAskPastHealthConditions(Action):

    def name(self) -> Text:
        return "action_ask_past_health_conditions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_past_health_conditions")
        return [SlotSet("last_question", "past_health_conditions")]

class ActionAskOtherDetails(Action):

    def name(self) -> Text:
        return "action_ask_other_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_ask_other_details")
        return [SlotSet("last_question", "other_details")]


class ActionProvideExplanation(Action):

    def name(self) -> Text:
        return "action_provide_explanation"
    
    

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_question = tracker.get_slot("last_question")

        if last_question == "symptoms":
            dispatcher.utter_message(response="utter_reason_for_asking_symptoms")
        elif last_question == "symptom_duration":
            dispatcher.utter_message(response="utter_reason_for_asking_duration")
        elif last_question == "symptom_severity":
            dispatcher.utter_message(response="utter_reason_for_asking_severity")
        elif last_question == "contact_info":
            dispatcher.utter_message(response="utter_reason_for_asking_contact")
        elif last_question == "past_health_conditions":
            dispatcher.utter_message(response="utter_reason_for_asking_past")
        else:
            dispatcher.utter_message(text="I'm not sure why this information is needed.")
            return []
        
        

        #Re-ask the last question after providing the explanation
        return [FollowupAction(f"action_ask_next_question")]   

# class ActionAskNextQuestion(Action):

#     def name(self) -> Text:
#         return "action_ask_next_question"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         symptom_list = tracker.get_slot("symptom_list")
#         if not symptom_list:
#             return [FollowupAction("action_ask_symptoms")]
#         symptom_duration = tracker.get_slot("symptom_duration")
#         if not symptom_duration:
#             return [FollowupAction("action_ask_symptom_duration")]
#         symptom_severity = tracker.get_slot("symptom_severity")
#         if not symptom_severity:
#             return [FollowupAction("action_ask_symptom_severity")]
#         contact_info = tracker.get_slot("contact_info")
#         if not contact_info:
#             return [FollowupAction("action_ask_contact_info")]
#         past_health_conditions = tracker.get_slot("past_health_conditions")
#         if not past_health_conditions:
#             return [FollowupAction("action_ask_past_health_conditions")]

#         return [FollowupAction("action_submit_and_provide_diagnosis")]
    

class ActionAskNextQuestion(Action):

    def name(self) -> Text:
        return "action_ask_next_question"

    def extract_entities(self, text: str) -> Dict[Text, Any]:
        tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [model.config.id2label[pred.item()] for pred in predictions[0]]

        return [label[2:] for label in labels if label != 'O']
    
    def find_age(self, message):
        digit_match = re.search(r'\b(1[0-1][0-9]|[1-9][0-9]?)\b', message)
        if digit_match:
            age = int(digit_match.group())
            if 1 <= age <= 110:
                return age
            

        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'twenty one': 21, 'twenty two': 22,
            'twenty three': 23, 'twenty four': 24, 'twenty five': 25,
            'twenty six': 26, 'twenty seven': 27, 'twenty eight': 28,
            'twenty nine': 29, 'thirty': 30, 'thirty one': 31, 'thirty two': 32,
            'thirty three': 33, 'thirty four': 34, 'thirty five': 35,
            'thirty six': 36, 'thirty seven': 37, 'thirty eight': 38,
            'thirty nine': 39, 'forty': 40, 'forty one': 41, 'forty two': 42,
            'forty three': 43, 'forty four': 44, 'forty five': 45,
            'forty six': 46, 'forty seven': 47, 'forty eight': 48,
            'forty nine': 49, 'fifty': 50, 'fifty one': 51, 'fifty two': 52,
            'fifty three': 53, 'fifty four': 54, 'fifty five': 55,
            'fifty six': 56, 'fifty seven': 57, 'fifty eight': 58,
            'fifty nine': 59, 'sixty': 60, 'sixty one': 61, 'sixty two': 62,
            'sixty three': 63, 'sixty four': 64, 'sixty five': 65,
            'sixty six': 66, 'sixty seven': 67, 'sixty eight': 68,
            'sixty nine': 69, 'seventy': 70, 'seventy one': 71, 'seventy two': 72,
            'seventy three': 73, 'seventy four': 74, 'seventy five': 75,
            'seventy six': 76, 'seventy seven': 77, 'seventy eight': 78,
            'seventy nine': 79, 'eighty': 80, 'eighty one': 81, 'eighty two': 82,
            'eighty three': 83, 'eighty four': 84, 'eighty five': 85,
            'eighty six': 86, 'eighty seven': 87, 'eighty eight': 88,
            'eighty nine': 89, 'ninety': 90, 'ninety one': 91, 'ninety two': 92,
            'ninety three': 93, 'ninety four': 94, 'ninety five': 95,
            'ninety six': 96, 'ninety seven': 97, 'ninety eight': 98,
            'ninety nine': 99, 'one hundred': 100, 'one hundred one': 101,
            'one hundred two': 102, 'one hundred three': 103, 'one hundred four': 104,
            'one hundred five': 105, 'one hundred six': 106, 'one hundred seven': 107,
            'one hundred eight': 108, 'one hundred nine': 109, 'one hundred ten': 110
        }

        words = message.lower().split()
    
        matched_numbers = []
        for i in range(len(words)):
            word = words[i]
            if word in number_words:
                matched_numbers.append(number_words[word])
            
            if i + 1 < len(words):
                phrase = ' '.join(words[i:i+2])
                if phrase in number_words:
                    matched_numbers.append(number_words[phrase])

        if(matched_numbers):
            return matched_numbers[-1]

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        latest_message = tracker.latest_message.get("text")
        last_question = tracker.get_slot("last_question")

        patient_responses = tracker.get_slot("patient_responses")
        if patient_responses is None:
            patient_responses = ""
       
        if latest_message:
            # Extract entities from the latest user message
            entities = self.extract_entities(latest_message)
            
            
            # Map entities to slots
            slot_updates = []
            if last_question == "symptoms":
                slot_updates.append(SlotSet("symptoms", latest_message))
                tracker.slots["symptoms"] = latest_message

            if 'SEVERITY' in entities:
                slot_updates.append(SlotSet("symptom_severity", True))
                tracker.slots["symptom_severity"] = True
            if 'DURATION' in entities or 'DOSAGE' in entities or 'DATE' in entities:
                slot_updates.append(SlotSet("symptom_duration", True))
                tracker.slots["symptom_duration"] = True
            age = self.find_age(latest_message)
            if last_question == 'patient_age' and age:
                slot_updates.append(SlotSet("patient_age", age))
                tracker.slots["patient_age"] = age
            words = ['woman', 'girl', 'lady', 'female', 'man', 'boy', 'guy', 'gentleman', 'male', 'non-binary', 'non binary', 'nonbinary', 'genderqueer', 'gender queer', 'genderfluid', 'gender fluid', 'trans']
            if 'SEX' in entities or any(word in latest_message.lower() for word in words):
                slot_updates.append(SlotSet("patient_sex", True))
                tracker.slots["patient_sex"] = True
    

            
            slot_updates.append(SlotSet("patient_responses", patient_responses +"|"+ latest_message))
            tracker.slots["patient_responses"] = (patient_responses + "|" + latest_message)

            # Optional: Add additional logic to handle specific cases

            # Notify user about the updates
            # if slot_updates:
            #     dispatcher.utter_message(text="Slots updated based on the latest message.")
            # else:
            #     dispatcher.utter_message(text="No relevant entities found in the latest message.")
       
        # for slot_update in slot_updates:
        #     tracker.slots[slot_update.key] = slot_update.value

        # Print slot values
        symptoms = tracker.get_slot("symptoms")
        symptom_duration = tracker.get_slot("symptom_duration")
        symptom_severity = tracker.get_slot("symptom_severity")
        age = tracker.get_slot("patient_age")
        sex = tracker.get_slot("patient_sex")
        contact_info = tracker.get_slot("contact_info")
        past_health_conditions = tracker.get_slot("past_health_conditions")
        patient_responses = tracker.get_slot("patient_responses")
        
        # Print slot values
        print(f"Symptoms: {symptoms}")
        print(f"Symptom Duration: {symptom_duration}")
        print(f"Symptom Severity: {symptom_severity}")
        print(f"Contact Info: {contact_info}")
        print(f"age: {age}")
        print(f"sex: {sex}")
        print(f"Past Health Conditions: {past_health_conditions}")
        print(f"responses {patient_responses}")
        
        symptoms = tracker.get_slot("symptoms")
        if not symptoms:
            return  slot_updates + [FollowupAction("action_ask_symptoms")] 
        symptom_duration = tracker.get_slot("symptom_duration")
        if not symptom_duration:
            return  slot_updates + [FollowupAction("action_ask_symptom_duration")] 
        symptom_severity = tracker.get_slot("symptom_severity")
        if not symptom_severity:
            return  slot_updates + [FollowupAction("action_ask_symptom_severity")] 
        patient_age = tracker.get_slot("patient_age")
        if not patient_age:
            return  slot_updates + [FollowupAction("action_ask_patient_age")]
        patient_sex = tracker.get_slot("patient_sex")
        if not patient_sex:
            return  slot_updates + [FollowupAction("action_ask_patient_sex")]
        contact_info = tracker.get_slot("contact_info")
        if not contact_info:
            return  slot_updates + [FollowupAction("action_ask_contact_info")]
        past_health_conditions = tracker.get_slot("past_health_conditions")
        if not past_health_conditions:
            return  slot_updates + [FollowupAction("action_ask_past_health_conditions")]
        other_details = tracker.get_slot("other_details")
        if not other_details:
            return  slot_updates + [FollowupAction("action_ask_other_details")]
        
        
            
        # else:
        #     dispatcher.utter_message(text="No message text found for slot update.")
        #     return [FollowupAction("action_submit_and_provide_diagnosis")] 

        return  slot_updates + [FollowupAction("action_submit_and_provide_diagnosis")]
    

class ActionAskNextQuestionHelper(Action):

    def name(self) -> Text:
        return "action_ask_next_question_helper"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        symptoms = tracker.get_slot("symptoms")
        if not symptoms:
            return  [FollowupAction("action_ask_symptoms")] 
        symptom_duration = tracker.get_slot("symptom_duration")
        if not symptom_duration:
            return  [FollowupAction("action_ask_symptom_duration")] 
        symptom_severity = tracker.get_slot("symptom_severity")
        if not symptom_severity:
            return  [FollowupAction("action_ask_symptom_severity")] 
        patient_age = tracker.get_slot("patient_age")
        if not patient_age:
            return  [FollowupAction("action_ask_patient_age")]
        patient_sex = tracker.get_slot("patient_sex")
        if not patient_sex:
            return  [FollowupAction("action_ask_patient_sex")]
        contact_info = tracker.get_slot("contact_info")
        if not contact_info:
            return  [FollowupAction("action_ask_contact_info")]
        past_health_conditions = tracker.get_slot("past_health_conditions")
        if not past_health_conditions:
            return  [FollowupAction("action_ask_past_health_conditions")]
        other_details = tracker.get_slot("other_details")
        if not other_details:
            return  [FollowupAction("action_ask_other_details")]
        
        return  [FollowupAction("action_submit_and_provide_diagnosis")]
    
# class ActionFillSymptomList(Action):

#     def name(self) -> Text:
#         return "action_fill_symptom_list"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         # Extract the current value of the shopping list slot
#         symptom_list = tracker.get_slot('symptom_list') or []
        
#         # Extract the latest user message and parse entities
#         latest_message = tracker.latest_message
#         entities = latest_message['entities']
        
#         # Extract groceries entities and add to the shopping list
#         new_symptoms = [entity['value'] for entity in entities if entity['entity'] == 'symptoms']
#         updated_symptom_list = symptom_list + new_symptoms
        
#         # Provide feedback to the user
#         if new_symptoms:
#             dispatcher.utter_message(text=f"Added to your sym list: {', '.join(new_symptoms)}")
#         else:
#             dispatcher.utter_message(text="I didn't catch any items to add to the sym list.")
        
#         # Return the updated shopping list slot
#         return [SlotSet('symptom_list', updated_symptom_list)]


class ActionUpdateSlots(Action):  
    def name(self) -> Text:
        return "action_update_slots"

    def extract_entities(self, text: str) -> Dict[Text, Any]:
        tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [model.config.id2label[pred.item()] for pred in predictions[0]]

        return [label[2:] for label in labels if label != 'O']


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        latest_message = tracker.latest_message.get("text")
        if latest_message:
            # Extract entities from the latest user message
            entities = self.extract_entities(latest_message)
            
            
            # Map entities to slots
            slot_updates = []
            if 'SEVERITY' in entities:
                slot_updates.append(SlotSet("symptom_severity", True))
            if 'DETAILED_DESCRIPTION' in entities or 'SIGN_SYMPTOM' in entities:
                slot_updates.append(SlotSet("symptoms", True))
            # if 'SEVERITY' in entities:
            #     slot_updates.append(SlotSet("severity", entities['SEVERITY']))
            if 'DURATION' in entities:
                slot_updates.append(SlotSet("symptom_duration", True))

            # Optional: Add additional logic to handle specific cases

            # Notify user about the updates
            if slot_updates:
                dispatcher.utter_message(text="Slots updated based on the latest message.")
            else:
                dispatcher.utter_message(text="No relevant entities found in the latest message.")

            return slot_updates
        else:
            dispatcher.utter_message(text="No message text found for slot update.")
            return []  

class ActionReaskLastQuestion(Action):

    def name(self) -> Text:
        return "action_reask_last_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_question = tracker.get_slot("last_question")

        if last_question == "symptoms":
            dispatcher.utter_message(response="action_ask_next_question")
        elif last_question == "symptom_duration":
            dispatcher.utter_message(response="action_ask_next_question")
        elif last_question == "symptom_severity":
            dispatcher.utter_message(response="action_ask_next_question")
        elif last_question == "past_health_conditions":
            dispatcher.utter_message(response="utter_ask_past_health_conditions")
        elif last_question == "contact_info":
            dispatcher.utter_message(response="utter_ask_contact_info")
        else:
            dispatcher.utter_message(text="I'm not sure which question to reask.")

        return []
    


class ActionReasonForAskingSymptoms(Action):

    def name(self) -> Text:
        return "action_reason_for_asking_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_reason_for_asking_symptoms")
        return [FollowupAction("action_reask_last_question")]

class ActionReasonForAskingSymptomDuration(Action):

    def name(self) -> Text:
        return "action_reason_for_asking_symptom_duration"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_reason_for_asking_duration")
        return [FollowupAction("action_reask_last_question")]

class ActionReasonForAskingSymptomSeverity(Action):

    def name(self) -> Text:
        return "action_reason_for_asking_symptom_severity"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_reason_for_asking_severity")
        return [FollowupAction("action_reask_last_question")]

class ActionReasonForAskingContactInfo(Action):

    def name(self) -> Text:
        return "action_reason_for_asking_contact_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_reason_for_asking_contact")
        return [FollowupAction("action_reask_last_question")]

class ActionReasonForAskingPastHealthConditions(Action):

    def name(self) -> Text:
        return "action_reason_for_asking_past_health_conditions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_reason_for_asking_past")
        return [FollowupAction("action_reask_last_question")]

# class ActionExtractMedicalEntities(Action):

#     def name(self) -> Text:
#         return "action_extract_medical_entities"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         # Initialize the NER pipeline
#         nlp = en_core_med7_trf.load()

#         # Get the latest user message
#         text = tracker.latest_message.get('text')

#         # Perform NER
#         doc = nlp(text)
#         entities = []
#         for ent in doc.ents:
#             entities.append({
#                 "start": ent.start_char,
#                 "end": ent.end_char,
#                 "value": ent.text,
#                 "entity": ent.label_
#             })

#         # Log the extracted entities
#         dispatcher.utter_message(text=f"Extracted entities: {entities}")

#         return []