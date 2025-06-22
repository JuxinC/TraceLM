import string
#nltk for NLP 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag  import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
import numpy as np
import pandas as pd
import time
#nltk.download('averaged_perceptron_tagger')


def preprocess(text, porterStemmer, cachedStopwords):
    
    string_text = str(text)
    #lowercase the string
    lower_case_string = string_text.lower()
    
    #Remove interpunction
    no_interpunction = lower_case_string.translate(str.maketrans('','',string.punctuation))
    
    #Remove numbers
    no_numbers = ''.join([i for i in no_interpunction if not i.isdigit()])
    
    #tokenize string
    tokens = word_tokenize(no_numbers)
    
    #remove stopwords
    tokens_without_sw = [word for word in tokens if not word in cachedStopwords]
    
    #Stem the tokens
    stemmedToken = list(map(porterStemmer.stem, tokens_without_sw))

    return(stemmedToken)

def preprocessNGrams(text, porterStemmer, cachedStopWords, nGramSize):
    string_text = str(text)
    
    #lowercase the string
    lower_case_string = string_text.lower()
    
    #Remove interpunction
    no_interpunction = lower_case_string.translate(str.maketrans('','',string.punctuation))
    
    #Remove numbers
    no_numbers = ''.join([i for i in no_interpunction if not i.isdigit()])
    
    #tokenize string
    tokens = word_tokenize(no_numbers)
    
    #Create the ngrams
    ngrams = list(nltk.ngrams(tokens, nGramSize))
    
    #remove all the n-grams containing a stopword
    cleanNGrams = [ngram for ngram in ngrams if not any(stop in ngram for stop in cachedStopWords)]
    
    #Stem the tokens
    stemmedNGrams = []
    for ngram in cleanNGrams:
        stemmed = list(map(porterStemmer.stem, ngram))
        stemmedNGrams.append(stemmed)
    return(stemmedNGrams)

#Function to transform date into a date object
def preprocess_jira_date(date_string):
    if(isinstance(date_string, str)):
        try:
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        return None
    
    
def findVerbs(tokenList):
    if not isinstance(tokenList, list):
        return []
    posTags = pos_tag(tokenList)
    verbAbrList = ['VBP', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']
    verbList = []
    for posTag in posTags:
        if posTag[1] in verbAbrList:
            verbList.append(posTag[0])
    return(verbList)

#Preprocess all the features and transform to the format needed for further processing.
def preprocessJiraData(cleanDataFrame, porterStemmer, cachedStopWords, startTime):
    
    nOfSteps = '3'

    #preprocess Summaries
    jira_summaries = cleanDataFrame['summary'].apply(lambda x: preprocess(x, porterStemmer, cachedStopWords))
    jira_summaries_2grams = cleanDataFrame['summary'].apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 2))
    jira_summaries_3grams = cleanDataFrame['summary'].apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 3))
    
    endTimeCleaningSummaries = time.time() - startTime
    print("1/" + nOfSteps + ") Finished Cleaning Summaries after " + str(endTimeCleaningSummaries) + " sec")

    #preprocess Descriptions
    cleanDataFrame.loc[:, 'description'] = cleanDataFrame['description'].fillna('')
    jira_descriptions = cleanDataFrame['description'].apply(lambda x: preprocess(x, porterStemmer, cachedStopWords))
    jira_descriptions_2grams = cleanDataFrame['description'].apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 2))
    jira_descriptions_3grams = cleanDataFrame['description'].apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 3))
    
    endTimeCleaningDescriptions = time.time() - startTime
    print("2/" + nOfSteps + ") Finished Cleaning Description after " + str(endTimeCleaningDescriptions) + " sec")

    #preprocess Dates
    jira_creation = cleanDataFrame['created_date'].apply(lambda x: preprocess_jira_date(x))
    jira_updated = cleanDataFrame['updated_date'].apply(lambda x: preprocess_jira_date(x))
    jira_resolved = cleanDataFrame['resolved_date'].apply(lambda x: preprocess_jira_date(x))
    endTimeCleaningDates = time.time() - startTime
    print("3/" + nOfSteps + ") Finished Cleaning Dates after " + str(endTimeCleaningDates) + " sec")

    
     #create JIRA corpus by merging Summary and Description
    jira_data = {'Issue_key_jira': cleanDataFrame['issue_id'], 
            'type':cleanDataFrame['type'],
            'assignee': cleanDataFrame['assignee'],
            'assignee_username':cleanDataFrame['assignee_username'],
            'reporter':cleanDataFrame['reporter'],
            'reporter_username':cleanDataFrame['reporter_username'],
            'Jira_created_date': jira_creation, 
            'Jira_updated_date': jira_updated, 
            'Jira_resolved_date': jira_resolved, 
            "summary":cleanDataFrame['summary'],
            'Summary': jira_summaries,
            'Summary_2grams': jira_summaries_2grams,
            'Summary_3grams': jira_summaries_3grams,
            "description":cleanDataFrame['description'],
            'Description': jira_descriptions,
            'Description_2grams': jira_descriptions_2grams,
            'Description_3grams': jira_descriptions_3grams,
            'jira_natual_text':cleanDataFrame['summary'] + cleanDataFrame['description'].astype(str),
            'Jira_natural_text': jira_summaries +  jira_descriptions,
            'Jira_natural_text_2grams': jira_summaries_2grams +  jira_descriptions_2grams,
            'Jira_natural_text_3grams': jira_summaries_3grams +  jira_descriptions_3grams}

    jira_processed_df = pd.DataFrame(data=jira_data)
    #Find verbs
    #jira_processed_df['verbs'] = jira_processed_df['Jira_natural_text'].apply(lambda x: findVerbs(x))
    
    return(jira_processed_df)

#Input dataframe and num of_comments, and bool to determine if comments need to be cleaned
def cleanJiraData(issue):
    startTime = time.time()

    #create an object of class PorterStemmer
    porterStemmer = PorterStemmer()
    
    #Find all stopwords
    cachedStopWords = stopwords.words("english")

    jira_issues_subset = issue[["issue_id", "type","assignee","assignee_username","reporter","reporter_username", "summary", "description", "created_date", "resolved_date", "updated_date"]]

    cleanedAndProcessedJiraData = preprocessJiraData(jira_issues_subset, porterStemmer = porterStemmer, cachedStopWords = cachedStopWords, startTime = startTime)
    return(cleanedAndProcessedJiraData)
