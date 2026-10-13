
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.tn.gov.in/"
LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Fields we want to collect from each scheme page.
FIELD_LABELS = [
    "Concerned Department",
    "Concerned District",
    "Organisation Name",
    "Scheme Title/Name",
    "Associated Scheme",
    "Sponsered By",
    "Funding Pattern",
    "Beneficiaries",
    "Types of Benefits",
    "Eligibility criteria",
    "Income",
    "Age From",
    "Age To",
    "Community",
    "How To avail",
    "Application is to be submitted to",
    "Validity of the Scheme",
    "Introduced On",
    "Description",
    "Scheme Type",
    "Uploaded File",
]

session = requests.Session()
session.headers.update(HEADERS)


def get_page(url):
    """Download a webpage and return its HTML."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def normalize(text):
    """Normalize labels for reliable matching."""
    return " ".join(text.split()).rstrip(":").strip().casefold()



def extract_scheme_details(url, listing_name):
    """Extract scheme fields, including multi-line values."""
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")

    lines = [
        line.strip()
        for line in soup.get_text("\n", strip=True).splitlines()
        if line.strip()
    ]

    known_labels = {normalize(label) for label in FIELD_LABELS}
    details = {}

    for label in FIELD_LABELS:
        target = normalize(label)
        value_lines = []

        for index, line in enumerate(lines):
            normalized_line = normalize(line)

            # Case 1: Label and value appear on the same line.
            if normalized_line.startswith(target + ":"):
                inline_value = line.split(":", 1)[1].strip()
                if inline_value:
                    value_lines.append(inline_value)

                # Capture any following lines until the next field.
                for next_line in lines[index + 1:]:
                    if normalize(next_line) in known_labels:
                        break
                    value_lines.append(next_line)

                break

            # Case 2: Label appears on a separate line.
            if normalized_line == target:
                for next_line in lines[index + 1:]:
                    if normalize(next_line) in known_labels:
                        break
                    value_lines.append(next_line)

                break

        details[label] = "\n".join(value_lines).strip()

    details["Scheme Title/Name"] = (
        details.get("Scheme Title/Name", "").strip()
        or listing_name
    )
    details["source_url"] = url

    return details


def main():
    print("Downloading scheme listing...")
    listing_html = get_page(LIST_URL)
    soup = BeautifulSoup(listing_html, "html.parser")

    # Collect unique scheme detail links from the listing.
    schemes = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        name = link.get_text(" ", strip=True)

        if "scheme_details.php" not in href or not name:
            continue

        scheme_url = urljoin(BASE_URL, href)

        if scheme_url in seen_urls:
            continue

        seen_urls.add(scheme_url)
        schemes.append({
            "name": name,
            "url": scheme_url
        })

    print(f"Found {len(schemes)} unique scheme links.")

    records = []
    failures = []

    for index, scheme in enumerate(schemes, start=1):
        print(
            f"[{index}/{len(schemes)}] "
            f"Collecting: {scheme['name']}"
        )

        try:
            record = extract_scheme_details(
                scheme["url"],
                scheme["name"]
            )
            records.append(record)

        except requests.RequestException as error:
            print(f"  Request failed: {error}")
            failures.append({
                "name": scheme["name"],
                "url": scheme["url"],
                "error": str(error)
            })

        except Exception as error:
            print(f"  Extraction failed: {error}")
            failures.append({
                "name": scheme["name"],
                "url": scheme["url"],
                "error": str(error)
            })

        # Avoid sending requests too rapidly.
        time.sleep(0.3)

    # Save scheme data and failures separately.
    with open("schemes.json", "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    with open("scrape_failures.json", "w", encoding="utf-8") as file:
        json.dump(failures, file, ensure_ascii=False, indent=2)

    print("\n--- SCRAPING SUMMARY ---")
    print(f"Scheme links found: {len(schemes)}")
    print(f"Successfully collected: {len(records)}")
    print(f"Failed pages: {len(failures)}")
    print("Saved data to schemes.json")
    print("Saved failures to scrape_failures.json")

    if records:
        print("\n--- FIRST COLLECTED RECORD ---")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
