*SEO Content Quality and Duplicate Detector*

This project is a machine learning pipeline developed to evaluate the SEO quality of web content and identify duplicate articles.
It was built as part of a Data Science assignment focusing on text analytics and content evaluation.

*Project Overview*

1. The system reads pre-scraped HTML data from a CSV file.
2. It extracts key text-based features such as word count, sentence count, and readability score.
3. It identifies important keywords using TF-IDF (Term Frequency–Inverse Document Frequency).
4. It detects duplicate or near-identical articles using cosine similarity (similarity threshold set at 0.80).
5. It flags "thin" content if the word count is below 500.
6. It classifies each article’s quality as Low, Medium, or High using a Random Forest model.

*Setup Instructions*

Clone the project repository:
git clone https://github.com/your-username/seo_assignment
cd seo_assignment
Install all required dependencies:
pip install -r requirements.txt
Run the notebook for analysis:
jupyter notebook notebooks/Test_final.ipynb

*Methodology and Key Decisions*

HTML content is parsed using BeautifulSoup, which removes script, style, and noscript tags.
Feature extraction includes word count, sentence count, and Flesch Reading Ease readability score.
TF-IDF is applied to extract the most significant words or phrases from each article.
Cosine similarity is used to detect duplicates; articles with a similarity above 0.80 are marked as duplicates.
Random Forest Classifier is chosen for its reliability with structured data and ability to show feature importance.

*Results Summary*

Total pages analyzed: 81
Duplicate pairs detected: 17
Thin content pages identified: 20
Model accuracy: 96%
Baseline word count accuracy: 80%
F1 Score (macro): 0.8571
Most important features for classification:
Readability (importance: 0.428)
Word count (importance: 0.320)
Sentence count (importance: 0.252)

*Limitations*

The quality labels used for training are rule-based and may not fully represent real human judgment of content quality.
Duplicate detection is limited to the dataset of approximately 80 articles and does not perform internet-wide comparison.
The readability metric (Flesch Reading Ease) may vary slightly for different text structures or formatting.