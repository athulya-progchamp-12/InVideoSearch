from utils.embeddings import get_bert_embedding
from sklearn.metrics.pairwise import cosine_similarity

def search_caption(query, captions):
    try:
        query_embedding = get_bert_embedding(query)
        captions_embeddings = [get_bert_embedding(caption) for caption in captions]
        similarities = [cosine_similarity(query_embedding, caption_embedding) for caption_embedding in captions_embeddings]
        best_match_idx = similarities.index(max(similarities))
        return captions[best_match_idx]
    except Exception as e:
        raise Exception(f"Error during caption search: {str(e)}")
