from fastapi import FastAPI, File, UploadFile,HTTPException
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
import uuid
import google.generativeai as gen_ai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
gemini_model = gen_ai.GenerativeModel("gemini-pro")



app = FastAPI()

directory = "D:\\pdf_extractdata"

# Initialize Qdrant client
qdrant_client = QdrantClient("http://localhost:6333")

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-t5-base')

# Create a collection in Qdrant
COLLECTION_NAME = "pdf_collection"
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE)
)

def split_text_into_chunks(text, chunk_size=2000):
    sentences = text.split('.')
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) + 1 <= chunk_size:
            chunk += sentence + '.'
        else:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sentence + '.'

    if chunk:
        chunks.append(chunk.strip())

    return chunks


@app.post("/combine-pdfs/")
async def combine_pdfs(files: list[UploadFile] = File(...)):
    # Define the path for the combined PDF file
    combined_pdf_location = os.path.join(directory, "combined.pdf")
    
    # Ensure the custom directory exists
    os.makedirs(os.path.dirname(combined_pdf_location), exist_ok=True)
    
    combined_pdf = fitz.open()
    
    for file in files:
        # Read the uploaded PDF file
        pdf_bytes = await file.read()
        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Append each page of the current PDF to the combined PDF
        for page_num in range(len(pdf_document)):
            
            combined_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    # Save the combined PDF to disk
    combined_pdf.save(combined_pdf_location)
    combined_pdf.close()
     # Extract text from the combined PDF
    combined_pdf = fitz.open(combined_pdf_location)
    full_text = ""
    for page_num in range(len(combined_pdf)):
        page = combined_pdf.load_page(page_num)
        full_text += page.get_text()
    
    # Split text into chunks
    text_chunks = split_text_into_chunks(full_text)
    
    # Generate embeddings for each chunk
    embeddings = model.encode(text_chunks)

    # Insert text chunks and their embeddings into Qdrant
    for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[rest.PointStruct(
                id=str(uuid.uuid4()),  # Generate a unique UUID for each chunk
                payload={"text": chunk},
                vector=embedding.tolist()   # Convert numpy array to list
            )]
        )
    
    # Print success message
    print(f"Combined PDF has been successfully saved as '{os.path.basename(combined_pdf_location)}'")
    
    # Return a JSON response with success message
    return JSONResponse(content={"message": f"Combined PDF has been successfully saved as '{os.path.basename(combined_pdf_location)}'"})


class QueryModel(BaseModel):
    query: str

@app.post("/search-similarity/")    
async def search_similarity(request: QueryModel):
    # Generate embedding for the query
    query_embedding = model.encode([request.query])[0]

    # Perform the search in Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=2  # Limit the search results to top 2 matches
    )

    
    # Extract the matched chunks from the search results
    results= [
        {
            "id": result.id,
            "text": result.payload["text"],
            "score": result.score
        }
        for result in search_result
    ]

    # Return the matched chunks as JSON response
    return JSONResponse(content={"matched chunks": results})


@app.post("/model/")    
async def search_similarity(request: QueryModel):
    # Generate embedding for the query
    query_embedding = model.encode([request.query])[0]

    # Perform the search in Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=2  # Limit the search results to top 2 matches
    )

    
    # Extract the matched chunks from the search results
    results= [
        {
            "id": result.id,
            "text": result.payload["text"],
            "score": result.score
        }
        for result in search_result
    ]

    context = " ".join([result["text"] for result in results])
    
    prompt= f""" I have a specific context extracted from a document. Please use this context to answer the question as thoroughly and accurately as possible. If the context does not explicitly provide the answer, offer a related answer based on the context. Ensure that your answer is directly connected to the information given in the context.:\n{context}\n
                    Question:\n{request.query}\n """
    
    try:
          response = gen_ai.generate_text(prompt=prompt)
          answer = response.result  # Extract the actual answer from the response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    return {"Response": answer}
    

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
