from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import os

app = FastAPI()

directory = "D:\\pdf_extractdata"

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
    
    # Print success message
    print(f"Combined PDF has been successfully saved as '{os.path.basename(combined_pdf_location)}'")
    
    # Return a JSON response with success message
    return JSONResponse(content={"message": f"Combined PDF has been successfully saved as '{os.path.basename(combined_pdf_location)}'"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
