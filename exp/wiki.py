import wikipediaapi
import json
user_agent = "DiagnoBot (somename77@gmail.com)"

def load_list_from_file(file_path):
    with open(file_path, 'r') as file:
        items = file.read().splitlines()
    return items

def fetch_wikipedia_introductions(page_titles):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')
    page_introductions = {}
    for title in page_titles:
        page = wiki_wiki.page(title)
        if page.exists():
        
            intro = page.summary
            page_introductions[title] = intro
        else:
             print(f"Page '{title}' does not exist on Wikipedia.")
    return page_introductions


def save_to_file(data, file_path):
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# medicine_names = load_list_from_file('medicine_name.txt')
# medical_conditions = load_list_from_file('medical_condition.txt')


# all_titles = medicine_names + medical_conditions
# wikipedia_contents = fetch_wikipedia_introductions(all_titles)


# save_to_file(wikipedia_contents, 'wikipedia_medical_data.json')
# print("saved to json!")

all_titles = load_list_from_file('additional_titles.txt')
def append_to_json(new_titles):
    
    with open('wikipedia_medical_data.json', 'r') as f:
        existing_data = json.load(f)
    
    # Fetch new data
    new_data = fetch_wikipedia_introductions(new_titles)
    
    # Update existing data with new data
    existing_data.update(new_data)
    
    # Save updated data back to the file
    with open('wikipedia_medical_data.json', 'w') as f:
        json.dump(existing_data, f, indent=4)



append_to_json(all_titles)