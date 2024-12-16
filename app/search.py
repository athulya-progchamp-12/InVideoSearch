from utils.embeddings import get_bert_embedding
from sklearn.metrics.pairwise import cosine_similarity

def search_caption(query, captions):
    """
    Search for the most relevant caption based on cosine similarity with the query.

    :param query: The user's search query.
    :param captions: The list of captions generated for the video keyframes.
    :return: The best matching caption.
    """
    try:
        query_embedding = get_bert_embedding(query)
        captions_embeddings = [get_bert_embedding(caption) for caption in captions]
        similarities = [cosine_similarity(query_embedding, caption_embedding) for caption_embedding in captions_embeddings]
        best_match_idx = similarities.index(max(similarities))
        return captions[best_match_idx]
    except Exception as e:
        raise Exception(f"Error during caption search: {str(e)}")

