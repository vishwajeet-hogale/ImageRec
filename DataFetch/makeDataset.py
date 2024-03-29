from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def connect_to_drive(gauth):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    return drive