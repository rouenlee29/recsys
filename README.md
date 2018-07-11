# recsys overview 

Files here are from Karpathy's [github repo](https://github.com/karpathy/arxiv-sanity-preserver), except for "arxivKeyword.py", which I have written to identify top k-number of keywords in each arxiv paper. Note that k is a parameter that could be determined by the user. 

# How to run "arxivKeyword.py"
1. Run fetch_papers.py (downloads arxiv papers in pdf format, you can specify the number of papers you want to download)
2. Run download_pdfs.py
3. Run parse_pdf_to_text.py
4. Run arxivKeyword.py. This will return a dictionary, where the key is paper name and values are keywords. 

For more info on steps 1-3, please see Karpathy's [github repo](https://github.com/karpathy/arxiv-sanity-preserver).

This can also be used to identify keywords for pdf documents. Just start from step 3, and remember to specify the correct directory for the "txt" folder. 
