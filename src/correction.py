import ollama

response = ollama.generate(
    model='qwen3.5:0.8b', 
    prompt='Give me a 3-word motivational phrase.'
)

# Extract and print the text response
print(response.response)