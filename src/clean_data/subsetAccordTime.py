import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def subsetAccordTime(commit_df,issue_df,link_df,interval):
    
    commit_start, commit_end = commit_df["Commit_date"].min(), commit_df["Commit_date"].max()
    issue_start, issue_end = issue_df["Jira_created_date"].min(), issue_df["Jira_created_date"].max()

    print(f"Commits Time Range: {commit_start} - {commit_end}")
    print(f"Issues Time Range: {issue_start} - {issue_end}")

    # Initial time interval
    current_start = '2012-01-02 15:10:14'#issue_start 2012-01-03T22:45:28Z
    current_start = datetime.strptime(current_start, '%Y-%m-%d %H:%M:%S')
    months = 1
    current_end = current_start + relativedelta(months=months)
    #last_valid_start = current_start
    #last_valid_end = current_end

    while current_end < commit_end:
        
        current_end = min(current_end, commit_end)
        # Filter data that matches the time interval
        commit_count = commit_df[(commit_df["Commit_date"] >= current_start) & 
                                (commit_df["Commit_date"] <= current_end)].shape[0]
        
        issue_count = issue_df[(issue_df["Jira_created_date"] >= current_start) & 
                            (issue_df["Jira_created_date"] <= current_end)].shape[0]
        
        print(f"Time Interval: {current_start} - {current_end}")
        print(f"Commit Count: {commit_count}, Issue Count: {issue_count}, Product: {commit_count * issue_count}")

        # Calculate whether commit × issue exceeds interval=1,000,000
        if commit_count * issue_count > interval:
            print("Exceeded the limit of 1,000,000 commit * issue combinations.")
            break  # Exit the loop, the current time interval is reasonable
        
        #last_valid_start = current_start
        #last_valid_end = current_end

        #Expand the time interval
        months += 1
        current_end = current_start + relativedelta(months=months)
        #current_start = current_end+step
    #current_start=last_valid_start
    #current_end=last_valid_end
    print(f"Finally selected time interval: {current_start} - {current_end}")
    
    # Filter commits
    filtered_commits = commit_df[(commit_df["Commit_date"] >= current_start) & 
                                (commit_df["Commit_date"] <= current_end)]
    print(filtered_commits.shape)
    # Filter issues
    filtered_issues = issue_df[(issue_df["Jira_created_date"] >= current_start) & 
                            (issue_df["Jira_created_date"] <= current_end)]
    print(filtered_issues.shape)
    # Get qualified commit_id and issue_id
    commit_ids = set(filtered_commits["commit_hash"])
    issue_ids = set(filtered_issues["Issue_key_jira"])

    # Filter change_set_link
    filtered_links = link_df[(link_df["commit_hash"].isin(commit_ids)) & 
                            (link_df["issue_id"].isin(issue_ids))]
    return filtered_commits,filtered_issues,filtered_links


    