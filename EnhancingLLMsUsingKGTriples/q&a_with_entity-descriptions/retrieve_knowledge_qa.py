from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from SPARQLWrapper import SPARQLWrapper, JSON

def load_qa_model():
    """Load the Question Answering model and tokenizer."""
    # Example usage: 'distilbert-base-uncased-distilled-squad'
    # A smaller, faster version of BERT that maintains a good balance between performance and resource efficiency.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def get_answer(qa_pipeline, question, context):
    """Get the answer from the QA model."""
    qa_result = qa_pipeline(question=question, context=context)
    return qa_result['answer']

def query_wikidata(entity):
    """Query Wikidata to get information about the entity."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item ?label "{entity}"@en.
      FILTER NOT EXISTS {{ ?item rdf:type wikibase:Statement. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

def print_wikidata_results(results):
    """Print the results from the Wikidata query."""
    seen_items = set()  # Set to keep track of printed items

    for result in results["results"]["bindings"]:
        item_value = result["item"]["value"]
        item_label = result["itemLabel"]["value"]

        if item_value not in seen_items:  # Check for duplicates
            print(item_value, item_label)
            seen_items.add(item_value)  # Add item to the seen set

if __name__ == "__main__":
    # Load the QA model
    qa_pipeline = load_qa_model()

    # Example question and context
    question = "Who is the president of the United States of America?"
    context = "Joe Biden is the current president of the United States of America."

    # Get the answer from the QA model
    answer = get_answer(qa_pipeline, question, context)
    print(f"Answer: {answer}")

    # Query Wikidata using the answer directly
    results = query_wikidata(answer)
    print_wikidata_results(results)