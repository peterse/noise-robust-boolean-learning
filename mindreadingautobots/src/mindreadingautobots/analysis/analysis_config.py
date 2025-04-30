import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import os

# Path to your service account JSON file
### PUT THE PATH TO THE SERVICE ACCOUNT JSON FILE HERE
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "velvety-tube-450516-r5-2dfa430c056c.json") 

# Define the scope (Google Sheets + Google Drive API)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",  # Allows access to files the service account has access to
    "https://www.googleapis.com/auth/drive.readonly"  # Read-only access to drive files
]
# Authenticate with Google
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# DO NOT CHANGE THE ORDEDR OF ANY OF THESE, ONLY APPEND
SPREADSHEET_NAMES = [
    "sparse_majority_41_5", 
    "sparse_majority_21_5", 
    "sparse_majority_51_3", 
    "sparse_majority_31_4",
    "sparse_parity_21_4",
    "hamilton_6_choose_6_nbits11_n5000"]

BF_VALS = [
    list(range(0, 50, 5)),
    list(range(0, 50, 5)),
    [0, 1.2, 3.0, 5.3, 8.0, 11.1, 14.8, 19.2, 24.6, 31.9],
    [0, 1.2, 3.0, 5.3, 8.0, 11.1, 14.8, 19.2, 24.6, 31.9],
    [0, 1, 2, 5, 7.5, 10, 12.5, 15, 17.5, 20],
    [0, 10, 20],

]
NAME_TEMPLATES = [
    '{}_sparse_majority_k5_nbits41_n2000_bf{}_seed1234',
    '{}_sparse_majority_k5_nbits21_n2000_bf{}_seed1234',
    '{}_sparse_majority_k3_nbits51_n2000_bf{}_seed1234_results',
    '{}_sparse_majority_k4_nbits31_n2000_bf{}_seed1234_results',
    '{}_sparse_parity_k4_nbits21_n5000_bf{}_seed1234',
    "{}_hamilton_6_choose_6_nbits11_n5000_bf{}_seed1234_results"
]
DATASET_NAMES = [
    r"$\text{MAJ}(40, 5)$",
    r"$\text{MAJ}(20, 5)$",
    r"$\text{MAJ}(50, 3)$",
    r"$\text{MAJ}(30, 4)$",
    r"$\text{PARITY}(20, 4)$",
    r"$\text{HAM}(6, 6)$",
    ]


def spreadsheet_to_dataframe(spreadsheet_name, bf_vals, sheetname_template, models=["RNN", "SAN"]):
    """Convert a google spreadsheet into a pandas dataframe.
    
    Args:
        spreadsheet_name: The title of the sheets file in drive
        bf_vals: The bitflip values to use; these must correspond to the bitflip values in the spreadsheet
        sheetname_template: The template for the sheetnames within the spreadsheet
    Returns:
        A pandas dataframe containing the data from the spreadsheet
    """

    # build a hash of the spreadsheets...
    spreadsheet = client.open(spreadsheet_name)
    sheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
    name_to_idx = {}
    for i, name in enumerate(sheet_names):
        name_to_idx[name] = i

    # compile everything into a big DF
    all_dfs = []
    for k in models:
        for bf in bf_vals:
            target = sheetname_template.format(k, bf)
            target_idx = name_to_idx.get(target)
            print("Loading:", target)
            sheet = spreadsheet.get_worksheet(target_idx)
            data = sheet.get_all_records()
            df = pd.DataFrame(data)

            # check for additional data
            try_1 = target + "_ud"
            try_2 = target + "_u"
            if try_1 in sheet_names or try_2 in sheet_names:
                if try_1 in sheet_names:
                    to_try = try_1
                else:
                    to_try = try_2

                print("Loading additional data for:", to_try)
                sheet_ud = spreadsheet.get_worksheet(name_to_idx[to_try])
                data_ud = sheet_ud.get_all_records()
                df_ud = pd.DataFrame(data_ud)
                # concatentate the above df with the new one
                df = pd.concat([df, df_ud], ignore_index=True)

            # tag this df with the bf value and keyword k
            df["bf"] = bf
            df["model"] = k
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    # save with the associated identifier
    return df