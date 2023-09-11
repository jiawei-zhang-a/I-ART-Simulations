import requests
import json

# Define the base URL for the ClinicalTrials.gov API
base_url = "https://clinicaltrials.gov/api/query/full_studies"

# Define the NCT ID for the study you're interested in
nct_id = "NCT01982942"  # Replace with the NCT ID of the study you want

# Define the parameters for the API request
params = {
    "expr": nct_id,
    "min_rnk": 1,
    "max_rnk": 1,
    "fmt": "json"
}

# Make the API request
response = requests.get(base_url, params=params)

# Print the status code
print(f"Status code: {response.status_code}")

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = json.loads(response.text)
    
    # Print the API response
    print(json.dumps(data, indent=4))
    
    try:
        # Extract the study details
        study_details = data['FullStudiesResponse']['FullStudies'][0]['Study']
        
        # Save the study details to a JSON file
        with open(f"{nct_id}_study_details.json", "w") as f:
            json.dump(study_details, f, indent=4)
            
        print(f"Study details saved to {nct_id}_study_details.json")
    except KeyError:
        print("Key 'FullStudies' not found in the API response.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
