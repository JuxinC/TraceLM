from datetime import datetime
import re
import pandas as pd
import string

#nltk for NLP 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')

#Function to transform natural text into unigram tokens
def preprocessNaturalLanguage(text, porterStemmer, cachedStopWords):
    string_text = str(text)
    #lowercase the string
    lower_case_string = string_text.lower()
    #Remove interpunction
    no_interpunction = lower_case_string.translate(str.maketrans('','',string.punctuation))
    #Remove numbers
    no_numbers = ''.join([i for i in no_interpunction if not i.isdigit()])
    #tokenize string
    tokens = word_tokenize(no_interpunction)
    #remove stopwords
    tokens_without_sw = [word for word in tokens if not word in cachedStopWords]
    #Stem the tokens
    stemmedToken = list(map(porterStemmer.stem, tokens_without_sw))
    return(stemmedToken)

#Function to transform natural text into n-gram tokens
def preprocessNGrams(text, porterStemmer, cachedStopWords, nGramSize):
    string_text = str(text)
    
    #lowercase the string
    lower_case_string = string_text.lower()
    
    #Remove interpunction
    no_interpunction = lower_case_string.translate(str.maketrans('','',string.punctuation))
    
    #Remove numbers
    no_numbers = ''.join([i for i in no_interpunction if not i.isdigit()])
    
    #tokenize string
    tokens = word_tokenize(no_interpunction)
    
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
def preprocessCommitDate(date_string):
    if isinstance(date_string, str):
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
    return None
    
#Remove the found Issue key from the log
def removeIssueKey(log_message):
    issue_keys = re.findall(r"\[?[A-Z]+\-[0-9]+\]?", log_message) #
    log_message_without_key = log_message
    for issue_key in issue_keys:
        log_message_without_key = log_message_without_key.replace(issue_key, "")
    return(log_message_without_key)

def unitNamesLambdaFunc(unitName, stemmer):
    #Lower case
    unitNameLowered = unitName.lower()
    
    #Remove interpunction
    noInterpunction = unitNameLowered.translate(str.maketrans('','',string.punctuation))
    
    #Remove numbers
    noNumbers = ''.join([i for i in noInterpunction if not i.isdigit()])
    
    stemmendUnitName = stemmer.stem(noInterpunction)
    
    
    return(stemmendUnitName)
    


#Method to clean all columns of the provided data
def cleanCommitData(commit_df): 
    #create an object of class PorterStemmer
    porterStemmer = PorterStemmer()
    
    #Find all stopwords
    cachedStopWords = stopwords.words("english")
    #Remove all revisions without an issue key in the log message
    #commit_df = rawCommitData[rawCommitData["related_issue_key"].notna()]

    #Execute cleaning methods on dataset
    processed_date_times = commit_df['committed_date'].apply(lambda x: preprocessCommitDate(x))
    cleaned_commit_logs = commit_df['message'].fillna('')
    cleaned_commit_logs = cleaned_commit_logs.apply(lambda x: removeIssueKey(x))
    processed_commit_logs = cleaned_commit_logs.apply(lambda x: preprocessNaturalLanguage(x, porterStemmer, cachedStopWords))
    processed_commit_logs_2grams = cleaned_commit_logs.apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 2))
    processed_commit_logs_3grams = cleaned_commit_logs.apply(lambda x: preprocessNGrams(x, porterStemmer, cachedStopWords, 3))
    
    #Put all data together into a new dataframe
    commit_data = {'commit_hash': commit_df["commit_hash"],
                'Author':commit_df["author"],
                'Email' : commit_df["author_email"],
                'Commit_date': processed_date_times,
                'Message':cleaned_commit_logs,
                #"Issue_key_commit": commit_df["related_issue_key"],
                'Logs': processed_commit_logs, 
                'Logs_2grams': processed_commit_logs_2grams, 
                'Logs_3grams': processed_commit_logs_3grams, 
                }
               
    commit_processed_df = pd.DataFrame(data=commit_data)

    return(commit_processed_df)