from qwikidata.linked_data_interface import get_entity_dict_from_api
import requests
import logging
import pandas as pd
import re
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from multiprocessing import Pool

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def search_missing_values(entity_id, lang='de'):
    """Fetch label and description from API for a given entity ID."""
    try:
        entity_dict = get_entity_dict_from_api(entity_id)
        label = entity_dict.get('labels', {}).get(lang, {}).get('value', 'Not found')
        description = entity_dict.get('descriptions', {}).get(lang, {}).get('value', 'Not found')
        return label, description
    except Exception as e:
        logging.error(f"Error retrieving entity {entity_id}: {e}")
        return 'Not found', 'Not found'

def extract_label_from_wikipedia(wikidata_id, lang='de'):
    """
    Get the title of a Wikipedia article using the entity's Wikidata ID to use it as label
    """
    try:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
        response = requests.get(url)
        data = response.json()

        if 'entities' in data and wikidata_id in data['entities']:
            entity = data['entities'][wikidata_id]
            if 'sitelinks' in entity and f"{lang}wiki" in entity['sitelinks']:
                article_url = entity['sitelinks'][f"{lang}wiki"]['url']
            else:
                return 'Not found'
        else:
            return 'Not found'

        # Fetch Wikipedia article content
        response = requests.get(article_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the first <h1> tag within the specified id in the HTML page and extract the title content
        title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
        return title

    except Exception as e:
        print(f"Error fetching Wikipedia title content: {e}")
        return 'Not found'

def extract_description_from_wikipedia(wikidata_id, lang='de'):
    """
    Get the description from Wikipedia for an Wikidata entity using its ID
    """
    try:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the response status code is not 200

        data = response.json()

        if 'entities' in data and wikidata_id in data['entities']:
            entity = data['entities'][wikidata_id]
            if 'sitelinks' in entity and f"{lang}wiki" in entity['sitelinks']:
                article_url = entity['sitelinks'][f"{lang}wiki"]['url']
            else:
                return 'Not found'
        else:
            return 'Not found'

        # Fetch Wikipedia article content
        response = requests.get(article_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the first <p> tag within the specified div in the HTML page and extract the text content
        try:
            first_paragraph = soup.find('div', {'id': 'mw-content-text'}).find('p', class_=False).text.strip()
        except Exception as e:
            logging.error(f"Error fetching Wikipedia content while using first option: {e}")
            # If the first approach fails, try an alternative approach
            first_paragraph = soup.find('p').text.strip()

        first_sentence = get_first_sentence(first_paragraph)
        cleaned_sentence = remove_itemization(first_sentence)
        return cleaned_sentence

    except Exception as e:
        logging.error(f"Error fetching Wikipedia description content: {e}")
        return 'Not found'

def get_first_sentence(article_content):
    # Define the regular expression pattern to match a sentence that includes a closing bracket
    pattern = r'^(.*?\))(.*?\.)'
    # Use re.match() to find the pattern
    match = re.match(pattern, article_content)
    if match:
        # If a closing bracket is found, return the (first) sentence
        return match.group(1) + match.group(2).rstrip('.')
    else:
        # If no closing bracket is found, return the first sentence up to the first period
        pattern = r'^(.*?\.)'
        match = re.match(pattern, article_content)
        return match.group(1).rstrip('.') if match else article_content

def remove_itemization(article_content):
    # Remove itemization using regular expressions
    cleaned_string = re.sub(r'\[\d+\]', '', article_content)
    return cleaned_string

def translate_text(text, target_lang='de'):
    """Translate text to the target language."""
    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        return translated
    except Exception as e:
        logging.error(f"Error translating text '{text}': {e}")
        return 'Not found'


def process_row(row):
    """Process a row to fill missing values."""
    skip_row = False
    for i in [1, 2, 4, 5]:
        if row[i] == 'Not found':
            # 1. Prioritize more reliable data from Wikidata
            for lang in ['de', 'en']:
                label, description = search_missing_values(row[0 if i < 3 else 3], lang=lang)
                if label != 'Not found' or description != 'Not found':
                    if label != 'Not found' and i in [1,4]:
                        row[i] = label
                    elif description != 'Not found' and i in [2,5]:
                        row[i] = description
                    if row[i] != 'Not found':
                        if lang == 'en':
                            row[i] = translate_text(row[i])
                        break
            if row[i] == 'Not found':
                for lang in ['de', 'en']:
                    if row[i] == 'Not found':
                        row[i] = extract_label_from_wikipedia(row[0 if i < 3 else 3],lang='lang') if i in [1, 4] else extract_description_from_wikipedia(row[0 if i < 3 else 3],lang='lang')
                        if row[i] != 'Not found':
                            if lang == 'en':
                                row[i] = translate_text(row[i])
                            break
            if row[i] == 'Not found':
                skip_row = True
    return row if not skip_row else None


def update_csv(input_csv_file, output_csv_file):
    """Update a CSV file by processing each row to fill missing values."""
    try:
        df = pd.read_csv(input_csv_file)
        headers = df.columns.tolist()

        with Pool() as pool:
            results = pool.map(process_row, df.values.tolist())

        df = pd.DataFrame([row for row in results if row is not None], columns=headers)
        df.to_csv(output_csv_file, index=False)
        logging.info(f"Updated CSV file '{output_csv_file}' created successfully!")
    except Exception as e:
        logging.error(f"Error updating CSV file: {e}")

if __name__ == "__main__":
    input_csv_file = 'entity2types.csv'
    output_csv_file = 'entity2types_2.0.csv'
    update_csv(input_csv_file, output_csv_file)
    print(f"Updated CSV file '{output_csv_file}' created successfully!")