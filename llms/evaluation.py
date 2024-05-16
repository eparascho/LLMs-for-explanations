import readability
import language_tool_python
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util


'''
This function calculates the coherence/relevance score between the query and response.
'''
def coherence_score(query, response):
    # create the embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embeddings.encode(query)
    response_embedding = embeddings.encode(response)
    # calculate similarity
    similarity = util.cos_sim(query_embedding, response_embedding)
    return similarity.item()


'''
This function calculates the grammatical errors in the response.
'''
def grammatical_errors(response):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(response)
    return len(matches)


'''
This function calculates the readability score of the response.
'''
def readability_score(response):
    results = readability.getmeasures(response, lang='en')
    return results['readability grades']['ARI']


'''
This function calculates the sentiment consistency between the query and response.
'''
def sentiment_consistency(query, response):
    query_sentiment = TextBlob(query).sentiment.polarity
    response_sentiment = TextBlob(response).sentiment.polarity
    return abs(query_sentiment - response_sentiment)


'''
This function calculates the concept coverage between the query and response.
'''
def concept_coverage(query, response):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([query, response])
    feature_names = vectorizer.get_feature_names_out()
    query_concepts = set([feature_names[i] for i in tfidf[0].nonzero()[1]])
    response_concepts = set([feature_names[i] for i in tfidf[1].nonzero()[1]])
    coverage = query_concepts.intersection(response_concepts)
    concepts = response_concepts.difference(query_concepts)
    return coverage, concepts


'''
This function prints all the structural quality evaluation metrics.
'''
def structural_quality_evaluation(query, response):
    coherence = coherence_score(query, response)
    grammatical = grammatical_errors(response)
    readability = readability_score(response)
    sentiment = sentiment_consistency(query, response)
    coverage, concepts = concept_coverage(query, response)
    print('Coherence/Relevance Score:', coherence)
    print('Number of Grammatical Errors:', grammatical)
    print('Automated Readability Index:', readability)
    print('Sentiment Consistency Score:', sentiment)
    print('Concepts Covered:', coverage)
    print('New Concepts Introduced:', concepts)
