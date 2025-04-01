import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa import retrieve_context

# ✅ Load the Fine-Tuned GPT-Neo Model
MODEL_PATH = "./fine_tuned_mistral"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

def generate_response(query, retrieved_docs, user_role):
    """Generates a response using the fine-tuned GPT-Neo model with context."""

    # ✅ Retrieve context
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."

    # ✅ Modify query based on user role
    role_prompt = {
        "Buyer": "You are an assistant helping an Amazon customer.",
        "Seller": "You are an Amazon seller expert providing accurate policy guidance.",
        "Admin": "You are an expert explaining Amazon policies in detail."
    }
    
    # ✅ Construct a well-structured prompt
    prompt = f"{role_prompt[user_role]}\n\nQuestion: {query}\nAnswer:"  # ❌ Removed 'Context' to prevent unnecessary output

    # ✅ Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # ✅ Fix max token length issue
    output = model.generate(
        **inputs, 
        max_length=min(len(inputs["input_ids"][0]) + 500, 2048),  # ✅ Controls only the generated response length
        temperature=0.7,  
        top_p=0.9,  
        no_repeat_ngram_size=3  
    )
    
    # ✅ Extract only the generated answer
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

    return generated_answer  # ✅ Only returns the answer, no context


# ✅ Test the function
if __name__ == "__main__":
    query = "How can I start selling on Amazon?"
    retrieved_docs = retrieve_context(query)
    print("🤖 GPT-Neo Response:", generate_response(query, retrieved_docs, "Seller"))
