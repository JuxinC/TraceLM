#Create a is_valid colomn. True if the trace link is valid.
def checkValidityTrace(issue_id, commit_id, link_set):
    # check if issue_key_jira occurs in the issue_list_commit
    return (issue_id, commit_id) in link_set