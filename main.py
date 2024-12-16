import pdfplumber
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
import time
import openai
from openai.error import RateLimitError  # Correct the import


# Set up OpenAI API key
openai.api_key = "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"
 
# Step 1: Extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Step 3: Create embeddings for the chunks
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained sentence transformer model
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Create FAISS index for storing embeddings
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance metric for similarity search
    index.add(embeddings)  # Add embeddings to the FAISS index
    return index

# Step 5: Handle a user's query by finding similar chunks
def handle_query(query, index, model, chunks, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query])
    # Perform similarity search
    D, I = index.search(np.array(query_embedding).astype(np.float32), k)  # D: distances, I: indices
    # Retrieve the corresponding chunks
    results = [chunks[i] for i in I[0]]
    return results

#Step 6: Generate a response using OpenAI's GPT model (Updated for new API)

def generate_response(query, context):
    prompt = f"Answer the following question based on the context: {query}\n\nContext: {context}"
    
    # Try to handle rate limiting gracefully
    retries = 5  # Number of retries before failing
    for attempt in range(retries):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if available and within quota
                prompt=prompt,
                max_tokens=500
            )
            return response['choices'][0]['text'].strip()  # Return the response if successful
        except RateLimitError as e:
            print(f"Rate limit exceeded, retrying... (Attempt {attempt+1}/{retries})")
            # Wait before retrying
            time.sleep(2 ** attempt)  # Exponential backoff (e.g., 1s, 2s, 4s, 8s...)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break  # Break out of loop if other errors occur
    return "Error: Could not get a response after retries."


# Step 7: Handle comparison queries by retrieving and comparing data
def handle_comparison_query(query, index, model, chunks, k=3):
    results = handle_query(query, index, model, chunks, k)
    # For simplicity, just combine the results for comparison
    comparison = "\n".join(results)
    return comparison

# Main function to integrate everything
def main():
    # List your PDF files here
    pdf_paths = ["C:/project1/data/file.pdf", "C:/project1/data/file2.pdf"]  # Update with your PDF file paths
    
    all_chunks = []

    # Step 1: Extract text and chunk PDFs
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    # Step 2: Create embeddings for all chunks
    embeddings = create_embeddings(all_chunks)
    embeddings = np.array(embeddings)

    # Step 3: Create FAISS index
    index = create_faiss_index(embeddings)

    # User query example
    user_query = "What are some historical examples of ratios and proportions used in geographic data analysis?"

    # Step 4: Handle user query and generate response
    print("\nHandling User Query...")
    retrieved_chunks = handle_query(user_query, index, SentenceTransformer('all-MiniLM-L6-v2'), all_chunks)
    context = "\n".join(retrieved_chunks)
    response = generate_response(user_query, context)
    print("\nResponse:", response)

    # For comparison queries
    comparison_response = handle_comparison_query(user_query, index, SentenceTransformer('all-MiniLM-L6-v2'), all_chunks)
    print("\nComparison Response:", comparison_response)

if __name__ == "__main__":
    main()