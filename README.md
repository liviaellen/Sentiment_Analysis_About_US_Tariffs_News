# Sentiment Analysis of US Tariffs News

This project performs sentiment analysis on news articles related to US tariffs using Natural Language Processing (NLP) techniques. It fetches recent news articles from NewsAPI, analyzes their sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner), and creates word embeddings using Word2Vec.

## Features

- Fetches news articles about US tariffs from NewsAPI
- Performs sentiment analysis using VADER
- Creates word embeddings using Word2Vec
- Visualizes word embeddings using t-SNE
- Saves results and visualizations to output directory

## Requirements

- Python 3.8+
- NewsAPI key (get one for free at https://newsapi.org/)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Sentiment_Analysis_About_US_Tariffs_News
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your NewsAPI key:
```
NEWS_API_KEY=your_api_key_here
```

## Usage

Run the script with default parameters:
```bash
python sentiment_analysis.py
```

### Command Line Arguments

- `--api-key`: NewsAPI key (optional if set in .env file)
- `--query`: Search query (default: "US tariffs")
- `--days`: Number of days to look back (default: 7)
- `--output-dir`: Output directory for results (default: "output")

Example with custom parameters:
```bash
python sentiment_analysis.py --query "US China trade" --days 14 --output-dir results
```

## Output

The script generates the following outputs in the specified output directory:

1. `news_articles.csv`: Contains the fetched articles with their sentiment scores
2. `word_embeddings.png`: Visualization of word embeddings using t-SNE

## Project Structure

- `sentiment_analysis.py`: Main script containing all functionality
- `requirements.txt`: List of required Python packages
- `.env`: Configuration file for API keys (not included in repository)
- `output/`: Directory containing generated results

## Dependencies

- requests: For making HTTP requests to NewsAPI
- pandas: For data manipulation
- nltk: For natural language processing
- gensim: For Word2Vec implementation
- scikit-learn: For t-SNE visualization
- matplotlib & seaborn: For data visualization
- python-dotenv: For loading environment variables

## License

This project is licensed under the MIT License - see the LICENSE file for details.
