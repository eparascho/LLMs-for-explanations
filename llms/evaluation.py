import json
import nltk
import numpy as np
import readability
import pandas as pd
from statistics import mean
import language_tool_python
from textblob import TextBlob
from scipy.stats import spearmanr
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
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

    # remove numbers, stopwrods and lowercse everything in query and response concepts
    query_concepts = set([c for c in query_concepts if not any(char.isdigit() for char in c)])
    response_concepts = set([c for c in response_concepts if not any(char.isdigit() for char in c)])
    query_concepts = set([c for c in query_concepts if c not in vectorizer.get_stop_words()])
    response_concepts = set([c for c in response_concepts if c not in vectorizer.get_stop_words()])
    query_concepts = set([c.lower() for c in query_concepts])
    response_concepts = set([c.lower() for c in response_concepts])

    # perfrom stemming and lemmatization in query and response concepts
    porter_stemmer = PorterStemmer()
    query_concepts = [porter_stemmer.stem(word) for word in query_concepts]
    response_concepts = [porter_stemmer.stem(word) for word in response_concepts]
    lemmatizer = WordNetLemmatizer()
    query_concepts = [lemmatizer.lemmatize(word) for word in query_concepts]
    response_concepts = [lemmatizer.lemmatize(word) for word in response_concepts]
    query_concepts = set(query_concepts)
    response_concepts = set(response_concepts)

    coverage = query_concepts.intersection(response_concepts)
    concepts = response_concepts.difference(query_concepts)

    return coverage, concepts, query_concepts, response_concepts


'''
This function prints all the structural quality evaluation metrics.
'''
def structural_quality_evaluation(query, response):
    coherence = coherence_score(query, response)
    grammatical = grammatical_errors(response)
    readability = readability_score(response)
    sentiment = sentiment_consistency(query, response)
    coverage, concepts, query_concepts, response_concepts = concept_coverage(query, response)
    final_coverage = len(coverage)/len(query_concepts)
    final_concepts = len(concepts)/len(response_concepts)

    return coherence, grammatical, readability, sentiment, final_coverage, final_concepts


'''
This function calculates the spearman rank correlation between the groundtruth and the LLM importances.
'''
def spearman_rank(df):
    spearman_corr, _ = spearmanr(df['rank_xai'], df['rank_llm'])

    return spearman_corr


'''
This function implements the dcg
'''
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


'''
This function implements the ndcg
'''
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


'''
This function calculates the NDCG between the groundtruth and the LLM importances.
'''
def ndcg(df):
    ndcg_xai = ndcg_at_k(df['rank_xai'], len(df))
    ndcg_llm = ndcg_at_k(df['rank_llm'], len(df))

    return abs(ndcg_xai - ndcg_llm)


'''
This function calculates the euclidean distance between the groundtruth and the LLM importances.
'''
def euclidean_dist(xai_response, llm_response):
    common_keys = set(xai_response.keys()).intersection(set(llm_response.keys()))
    lime_values = [xai_response[key] for key in common_keys]
    llm_values = [llm_response[key] for key in common_keys]
    
    return euclidean(lime_values, llm_values)


'''
This function calculates the content quality evaluation metric having as groundtruth the results of the respective XAI method.
'''
def content_xai_quality_evaluation(instance_interpret, llm_response):

    # load the explanation
    with open(f'../data/explainability_output/local_lime_{instance_interpret}.json') as f:
        xai_response = json.load(f)   

    # processing of the data
    scaler = MinMaxScaler()
    df_xai = pd.DataFrame(list(xai_response.items()), columns=['feature', 'value'])
    df_xai['value'] = scaler.fit_transform(df_xai[['value']])
    df_llm = pd.DataFrame(list(llm_response.items()), columns=['feature', 'value'])
    df_llm['value'] = scaler.fit_transform(df_llm[['value']])
    df_merged = pd.merge(df_xai, df_llm, on='feature', suffixes=('_xai', '_llm'))  # it takes only the similar features
    df_merged['rank_xai'] = df_merged['value_xai'].rank()
    df_merged['rank_llm'] = df_merged['value_llm'].rank()

    # compute and print the metrics
    spearman_corr = spearman_rank(df_merged)
    ndcg_dif = ndcg(df_merged)
    eucl_dist = euclidean_dist(xai_response, llm_response)

    return spearman_corr, ndcg_dif, eucl_dist


'''
This function appends in the corresponding lists the evaluation metrics for each instance.
'''
def instance_evaluation(coherences, grammaticals, readabilities, sentiments, conc_covs, conc_intrs, spearman_corrs, ndcg_difs, eucl_dists, prompt, user_response, instance_interpret, developer_response):
    coherence, grammatical, readability, sentiment, conc_cov, conc_intr = structural_quality_evaluation(prompt, user_response)
    coherences.append(coherence)
    grammaticals.append(grammatical)
    readabilities.append(readability)
    sentiments.append(sentiment)
    conc_covs.append(conc_cov)
    conc_intrs.append(conc_intr)
    spearman_corr, ndcg_dif, eucl_dist = content_xai_quality_evaluation(instance_interpret, developer_response)
    spearman_corrs.append(spearman_corr)
    ndcg_difs.append(ndcg_dif)
    eucl_dists.append(eucl_dist)

    return coherences, grammaticals, readabilities, sentiments, conc_covs, conc_intrs, spearman_corrs, ndcg_difs, eucl_dists


'''
This function prints all the aggregated evaluation metrics.
'''
def aggregated_evaluation(model, learning, coherences, grammaticals, readabilities, sentiments, conc_covs, conc_intrs, spearman_corrs, ndcg_difs, eucl_dists):
    print(model, ' in ', learning, '-shot learning: ')
    coherences = [x for x in coherences if str(x) != 'nan']
    print('Avg coherence:', mean(coherences))
    grammaticals = [x for x in grammaticals if str(x) != 'nan']
    print('Avg number of grammatical errors:', mean(grammaticals))
    readabilities = [x for x in readabilities if str(x) != 'nan']
    print('Avg ARI:', mean(readabilities))
    sentiments = [x for x in sentiments if str(x) != 'nan']
    print('Avg sentiment consistency:', mean(sentiments))
    conc_covs = [x for x in conc_covs if str(x) != 'nan']
    print('Avg percentage of concepts covered:', mean(conc_covs))
    conc_intrs = [x for x in conc_intrs if str(x) != 'nan']
    print('Avg percentage of new concepts introduced:', mean(conc_intrs))
    spearman_corrs = [x for x in spearman_corrs if str(x) != 'nan']
    print('Avg spearman rank correlation:', mean(spearman_corrs))
    ndcg_difs = [x for x in ndcg_difs if str(x) != 'nan']
    print('Avg NDCG differences:', mean(ndcg_difs))
    eucl_dists = [x for x in eucl_dists if str(x) != 'nan']
    print('Avg euclidean distances:', mean(eucl_dists))