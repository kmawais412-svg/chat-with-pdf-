import time
import fitz
from backend import ocr_tesseract

pdf_path = "sample.pdf"
MAX_PAGES = 3


def make_sample_pdf(original_path, max_pages, output_path="temp_sample.pdf"):
    doc = fitz.open(original_path)
    new_doc = fitz.open()
    for i in range(min(max_pages, len(doc))):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    return output_path


test_pdf_path = make_sample_pdf(pdf_path, MAX_PAGES)
print(f"✅ Test ke liye chota PDF bana: {test_pdf_path} ({MAX_PAGES} pages)\n")

print("▶️ Tesseract chal raha hai...")

start_time = time.time()
result = ocr_tesseract(test_pdf_path)
end_time = time.time()
response_time = end_time - start_time

total_chars = sum(len(page.page_content) for page in result)
total_words = sum(len(page.page_content.split()) for page in result)

print(f"⏱️  Response Time: {response_time:.2f} seconds")
print(f"📄 Pages tested: {len(result)}")
print(f"🔤 Characters: {total_chars}")
print(f"📝 Words: {total_words}")
print(f"\n--- Page 1 Preview ---")
print(result[0].page_content[:300])