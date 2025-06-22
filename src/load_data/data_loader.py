import sqlite3
import pandas as pd
import os
import re
def load_data_from_db(db_path):
    """Load data from SQLite database"""
    print(db_path)
    connection = sqlite3.connect(db_path)
    # Load the 'changeset' (commit) table
    changeset_query = "SELECT * FROM change_set WHERE commit_hash IS NOT NULL"
    changeset_df = pd.read_sql(changeset_query, connection)
    
    # Load the 'issue' table
    issue_query = "SELECT * FROM issue WHERE issue_id IS NOT NULL"
    issue_df = pd.read_sql(issue_query, connection)

    # Load the 'link' table
    link_query = "SELECT * FROM change_set_link WHERE issue_id IS NOT NULL and commit_hash IS NOT NULL"
    link_df = pd.read_sql(link_query, connection)
    
    connection.close()
    return changeset_df, issue_df, link_df

def get_all_sqlite_files(folder_path):
    """Get all SQLite files in the specified folder"""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.sqlite3')]

def remove_illegal_chars(df):
    # Excel except '\t', '\n', '\r'
    illegal_char_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def clean_cell(cell):
        if isinstance(cell, str):
            return illegal_char_pattern.sub('', cell)
        return cell

    return df.applymap(clean_cell)
