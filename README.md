
# YouTube Video Recommendation System Using BERT Embeddings

## Project Overview
This project builds a content-based video recommendation system for YouTube trending videos using semantic similarity of video titles. It leverages the pre-trained BERT language model to generate dense vector embeddings of video titles and uses cosine similarity to recommend videos with similar content.

## Dataset
- Source: [Kaggle - Top Trending Videos YouTube 2021](https://www.kaggle.com/jyotmakadiya/top-trending-videos-youtube-2021)
- Data file: `US_videos_data.csv`
- Selected Columns:
  - `title`: Video title
  - `channelTitle`: Name of the YouTube channel
  - `likes`: Number of likes
  - `dislikes`: Number of dislikes
  - `thumbnail_link`: URL to video thumbnail image
  - `description`: Video description

## Environment Setup
- Python 3.x
- Key Libraries:
  - `pandas`, `numpy` for data manipulation
  - `tensorflow` and `transformers` for BERT embedding extraction
  - `scikit-learn` for cosine similarity
  - `re` for text cleaning

Install dependencies via:

```bash
pip install pandas numpy tensorflow transformers scikit-learn
```

## Step-by-Step Workflow

### 1. Data Loading and Cleaning
- Load dataset using pandas.
- Select relevant columns.
- Remove duplicate entries.
- Clean the video titles by removing special characters and converting to lowercase.

### 2. Embedding Extraction with BERT
- Load the pre-trained BERT tokenizer and model (`bert-base-uncased`) from Hugging Face.
- Tokenize video titles and generate embeddings using BERT’s pooled output layer.
- Batch processing is used for efficient embedding generation.
- Store embeddings as a new column in the DataFrame.

### 3. Save & Load Processed Data
- Save the DataFrame with embeddings using pandas' pickle serialization.
- Reload the saved DataFrame as needed to avoid recomputing embeddings.

### 4. Recommendation Function
- Input: A video title query string.
- Clean the input text using the same cleaning function.
- Generate BERT embedding for the input title.
- Compute cosine similarity between the input embedding and all stored embeddings.
- Sort videos by similarity and return the top N recommendations excluding the query video.

### 5. Display Recommendations
- Display a neat table of recommended videos including:
  - Title
  - Channel name
  - Likes & dislikes count
  - Similarity score
- Display thumbnails of recommended videos.

## Usage Example

```python
input_title = "Speak english fluently"
recommendations = recommend_videos(input_title, df, top_n=5)
print(recommendations[['title', 'channelTitle', 'likes', 'dislikes', 'similarity']])
```

## Notes

- The system relies on the semantic meaning of titles, thus it can recommend videos even if exact keywords differ.
- Embeddings are generated on GPU if available, speeding up the process.
- The cosine similarity metric measures closeness between embeddings.

## Future Improvements

- Extend similarity calculations to include video descriptions.
- Add user feedback loop for personalized recommendations.
- Implement a web interface for interactive recommendations.

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Transformers Library by Hugging Face](https://huggingface.co/transformers/)
- [Kaggle Dataset](https://www.kaggle.com/jyotmakadiya/top-trending-videos-youtube-2021)
