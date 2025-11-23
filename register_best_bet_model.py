"""
Register best bet model with Vertex AI
"""
from google.cloud import aiplatform
import json

PROJECT_ID = "elite-hangar-479017-m8"
REGION = "us-central1"
MODEL_URI = "gs://parlaydesk-ml-models/models/best-bet-finder-v1"
SKLEARN_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"

print("📝 Registering Best Bet Finder model with Vertex AI...\n")

aiplatform.init(project=PROJECT_ID, location=REGION)

model = aiplatform.Model.upload(
    display_name="best-bet-finder-production",
    artifact_uri=MODEL_URI,
    serving_container_image_uri=SKLEARN_CONTAINER,
    description="Multi-model system: Win prob, Spread cover, Over/Under"
)

print(f"✅ Model registered!")
print(f"Name: {model.display_name}")
print(f"Resource: {model.resource_name}\n")

model_info = {
    "project_id": PROJECT_ID,
    "model_id": model.name,
    "model_resource_name": model.resource_name,
    "region": REGION
}

with open('vertex_model_info_best_bet.json', 'w') as f:
    json.dump(model_info, f, indent=2)
    
print("✅ Saved: vertex_model_info_best_bet.json")
print("\nNext: Deploy with deploy_to_endpoint_FIXED.py")
