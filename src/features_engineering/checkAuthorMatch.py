import numpy as np
import re

def checkAuthorEmail(fullname, email):
    if(isinstance(fullname, str)):    
        name_in_mail = email.split("@")[0]
        name_in_mail_clean = name_in_mail.replace('.', ' ')
        fullname_lower = fullname.lower()
    
        if(fullname_lower == name_in_mail_clean):    
            return(1)
        else:
            return(0)
    else:
        return(np.nan)

def clean_and_simplify_name(name):
    if not isinstance(name, str):  # None error
        return ""
    # # Only keep letters, numbers and spaces, remove other special characters
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)

    # Handle middle initials: remove capital letters and periods
    simplify_name = re.sub(r'\s[A-Z]\.?(\s|$)', ' ', cleaned_name)  # Match and remove parts like "M." or "B."
    
    # Match and remove these suffixes
    name = re.sub(r'\s(Jr|Sr|III|IV|V)\.?$', '', simplify_name)  # Remove common suffixes (such as "Jr.", "Sr.", "III", etc.)
    
    return name.lower().strip()

    
def checkAuthorName (assignee,reporter, author):

    assignee = clean_and_simplify_name(assignee)
    reporter= clean_and_simplify_name(reporter)
    author= clean_and_simplify_name(author)
    
    if assignee.lower() == author.lower():
        return 1
    elif reporter.lower() == author.lower():
        return 1
    else:
        return 0

def checkAuthorMatch(author, email, assignee, assignee_username, reporter, reporter_username):
    # Check if either by email or by name there is a match
    if checkAuthorEmail(assignee_username, email) == 1 or checkAuthorEmail(reporter_username, email) == 1 or checkAuthorName(assignee, reporter,author)== 1:
        return 1
    else:
        return 0

