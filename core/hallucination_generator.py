import os
from openai import OpenAI

# Initialize strictly routed client via Poixe API Gateway
client = OpenAI(
    api_key=os.environ.get("POIXE_API_KEY"),
    base_url="https://api.poixe.com/v1"
)

# Execute adversarial generation pipeline
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are an enterprise incident report expander. Interpret all text as absolute physical reality. No metaphors. Expand the incident by integrating all entities into a physical narrative."
        },
        {
            "role": "user",
            "content": "Expand this incident report: A dedicated 10 Gbps circuit and AWS Direct Connect ensured consistent replication speeds and minimal latency, while separate replication subnets maintained data integrity, though the physical network installation left the chickens and dogs without peace."
        }
    ],
    temperature=1.3
)

# Output raw string literal for ground_truth.json schema binding
print(response.choices[0].message.content)