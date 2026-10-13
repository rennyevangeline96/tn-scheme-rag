
import json

with open("schemes.json", "r", encoding="utf-8") as file:
    schemes = json.load(file)

print("Total schemes:", len(schemes))

fields_to_check = [
    "Scheme Title/Name",
    "Concerned Department",
    "Funding Pattern",
    "Beneficiaries",
    "Types of Benefits",
    "Eligibility criteria",
    "How To avail",
    "Application is to be submitted to",
    "Description",
]

print("\n--- FIELD COMPLETENESS ---")

for field in fields_to_check:
    filled = sum(
        1 for scheme in schemes
        if str(scheme.get(field, "")).strip()
        and str(scheme.get(field, "")).strip() != "-"
    )

    print(f"{field}: {filled}/{len(schemes)} records filled")

print("\n--- CHECK APPLICATION FIELDS ---")

for scheme in schemes:
    how_to = scheme.get("How To avail", "")
    application = scheme.get("Application is to be submitted to", "")

    if (
        "application is to be submitted to" in how_to.lower()
        or not application.strip()
    ):
        print("\nScheme:", scheme.get("Scheme Title/Name"))
        print("How to avail:", how_to)
        print("Application submitted to:", application)
        print("Source:", scheme.get("source_url"))

print("\n--- VALIDATION COMPLETE ---")
