import pymupdf

pdf_path = "Indian_Polity_Laxmi-Kant-6th-Edition-.pdf"
doc = pymupdf.open(pdf_path)

for page_number, page in enumerate(doc):
    text = page.get_text().strip()

    if "CONTENTS" in text.upper():
        print(f"\n========== PDF PAGE {page_number + 1} ==========")
        print(text[:3000])