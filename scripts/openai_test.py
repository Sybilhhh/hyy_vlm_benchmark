# import os
# import openai
# from openai import AzureOpenAI
# import urllib.request
# import ssl
# # Create an SSL context that skips verification
# ssl._create_default_https_context = ssl._create_unverified_context
# openai.verify_ssl_certs = False

# # endpoint = "https://openaieastus2instance.openai.azure.com/"
# endpoint = "https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview"
# model_name = "gpt-4o"
# deployment = "gpt-4o-cv-chx"

# subscription_key = "990a353da7b44bef8466402378c486cd"
# api_version = "2024-12-01-preview"

# client = AzureOpenAI(
#     api_version=api_version,
#     azure_endpoint=endpoint,
#     api_key=subscription_key
# )

# response = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant.",
#         },
#         {
#             "role": "user",
#             "content": "I am going to Paris, what should I see?",
#         }
#     ],
#     max_tokens=4096,
#     temperature=1.0,
#     top_p=1.0,
#     model=model_name
# )

# print(response.choices[0].message.content)

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="990a353da7b44bef8466402378c486cd",
    api_version="2024-12-01-preview",
    azure_endpoint="https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview"
)

try:
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("API connection successful")
except Exception as e:
    print(f"API connection failed: {e}")
    