import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

GOOGLE_APPLICATION_CREDENTIALS = './secret/credentials.json'

def load_data_from_google_spreadsheet(spreadsheet_id, worksheet_name, credentials_file=GOOGLE_APPLICATION_CREDENTIALS):
    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # Authenticate using credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    # Open the specified spreadsheet
    sheet = client.open_by_key(spreadsheet_id)
    worksheet = sheet.worksheet(worksheet_name)

    # Get all values from the worksheet
    rows = worksheet.get_all_values()

    # Convert data to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])
    
    return df
