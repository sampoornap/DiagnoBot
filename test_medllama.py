from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# model_name = "ruslanmv/Medical-Llama3-8B"
# device_map = 'auto'
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,)
# model = AutoModelForCausalLM.from_pretrained( model_name,quantization_config=bnb_config, trust_remote_code=True,use_cache=False,device_map=device_map)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

model_name = "ruslanmv/Medical-Llama3-8B"
device_map = 'cpu'  # Use CPU
bnb_config = BitsAndBytesConfig(load_in_4bit=False)  # Disable 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
def askme(question):
    sys_message = ''' 
    You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
    provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    '''   
    # Create messages structured for the chat template
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": question}]
    
    # Applying chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
    
    # Extract and return the generated text, removing the prompt
    response_text = tokenizer.batch_decode(outputs)[0].strip()
    answer = response_text.split('<|im_start|>assistant')[-1].strip()
    return answer
# Example usage
# - Context: First describe your problem.
# - Question: Then make the question.

question = '''I'm a 35-year-old male and for the past few months, I've been experiencing fatigue, 
increased sensitivity to cold, and dry, itchy skin. 
Could these symptoms be related to hypothyroidism? 
If so, what steps should I take to get a proper diagnosis and discuss treatment options?'''

print(askme(question))
