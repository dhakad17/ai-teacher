import pymupdf
import re

PDF_PATH = "Indian_Polity_Laxmi-Kant-6th-Edition-.pdf"

doc = pymupdf.open(PDF_PATH)

# We know the main Table of Contents is around PDF page 23.
TOC_PAGE = 23

text = doc[TOC_PAGE - 1].get_text()

chapters = []

for line in text.splitlines():

    line = line.strip()

    # Examples:
    # 1Historical Background
    # 2Making of the Constitution
    # 3Salient Features of the Constitution

    match = re.match(r"^(\d{1,2})\s*(.+)$", line)

    if match:
        number = int(match.group(1))
        title = match.group(2).strip()

        chapters.append({
            "chapter_number": number,
            "title": title
        })


print("\n========== CHAPTERS FROM TOC ==========\n")

for chapter in chapters:
    print(
        f"Chapter {chapter['chapter_number']}: "
        f"{chapter['title']}"
    )

print("\nTotal chapters found:", len(chapters))