import os
import requests
from pathlib import Path
from dotenv import load_dotenv
import msal

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TEAM_ID = os.getenv("TEAM_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")
DOWNLOAD_DIR = Path(os.getenv("TEAMS_DOWNLOAD_DIR", "teams_files"))

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def get_access_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET
    )

    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )

    if "access_token" not in result:
        raise RuntimeError(f"Token error: {result}")

    return result["access_token"]


def graph_get(url, token):
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()
    return response.json()


def graph_download(url, token, output_path):
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)


def get_channel_files_folder(token):
    url = f"{GRAPH_BASE}/teams/{TEAM_ID}/channels/{CHANNEL_ID}/filesFolder"
    return graph_get(url, token)


def list_folder_children(token, drive_id, item_id):
    url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/children"
    return graph_get(url, token).get("value", [])


def download_pdf_files_from_teams():
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    token = get_access_token()

    folder = get_channel_files_folder(token)
    drive_id = folder["parentReference"]["driveId"]
    folder_id = folder["id"]

    files = list_folder_children(token, drive_id, folder_id)

    downloaded_files = []

    for item in files:
        name = item["name"]

        if not name.lower().endswith(".pdf"):
            continue

        item_id = item["id"]
        output_path = DOWNLOAD_DIR / name

        download_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/content"

        print(f"Κατέβασμα: {name}")
        graph_download(download_url, token, output_path)

        downloaded_files.append(str(output_path))

    return downloaded_files