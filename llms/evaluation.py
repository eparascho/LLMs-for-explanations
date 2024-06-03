import math
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
    return results['readability grades']['FleschReadingEase']


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
    return coverage, concepts, len(query_concepts), len(response_concepts)


'''
This function prints all the structural quality evaluation metrics.
'''
def structural_quality_evaluation(query, response):
    coherence = coherence_score(query, response)
    grammatical = grammatical_errors(response)
    readability = readability_score(response)
    sentiment = sentiment_consistency(query, response)
    coverage, concepts, query_concepts, response_concepts = concept_coverage(query, response)
    print('Coherence/Relevance Score:', coherence)
    print('Number of Grammatical Errors:', grammatical)
    print('Flesch Reading Ease:', readability)
    print('Sentiment Consistency Score:', sentiment)
    print('Percentage of concepts covered:', len(coverage)/query_concepts)
    print('Concepts Covered:', coverage)
    print('Percentage of new concepts introduced:', len(concepts)/response_concepts)
    print('New Concepts Introduced:', concepts)


'''
This function calculates the content quality evaluation metric having as groundtruth the results of the respective XAI method.
'''
def content_xai_quality_evaluation(xai_response, llm_response):
    # Calculate Euclidean distance
    distance = 0.0
    for key in xai_response:
        if key in llm_response:
            distance += (xai_response[key] - llm_response[key]) ** 2
        else:
            distance += xai_response[key] ** 2
    print('Relative distance between global LIME feature importance vs. LLM feature importance', math.sqrt(distance))
