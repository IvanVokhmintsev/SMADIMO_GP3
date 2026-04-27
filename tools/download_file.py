from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

from langchain.tools import tool

def make_download_tools(state):
    @tool
    def download_file_from_url(url: str, output_dir: str = "data") -> dict:
        """
        Download a file from a direct URL and save it locally.

        Use this tool when the agent needs to load a dataset file by link.
        The tool returns local file metadata so the agent can pass the path
        to the next data-processing tools.

        Args:
            url: Direct URL to the dataset file.
            output_dir: Local directory where the file should be saved.

        Returns:
            Dictionary with download status, local file path, file name and size.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)


        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()


            file_name = None
            content_disposition = response.headers.get("Content-Disposition")

            if content_disposition and "filename=" in content_disposition:
                file_name = content_disposition.split("filename=")[-1].strip("\"' ")

            if not file_name:
                file_name = 'dataset.csv'

            file_name = unquote(file_name)
            local_file_path = output_path / file_name

            with open(local_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            state.df = pd.read_csv(local_file_path)

            return {
                "status": "success",
                "file_path": str(local_file_path),
                "file_name": local_file_path.name,
                "url": url,
            }

        except requests.exceptions.RequestException as error:
            return {
                "status": "error",
                "message": f"Download failed: {error}",
                "url": url,
            }
    return [download_file_from_url]