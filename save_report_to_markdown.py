from datetime import datetime
from pathlib import Path

def save_report_to_markdown(text: str) -> str:
    """Save text report to a markdown file with timestamped name in results/ folder."""

    # create folder if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # file path
    file_path = output_dir / f"result_{timestamp}.md"

    # write markdown file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return str(file_path)
