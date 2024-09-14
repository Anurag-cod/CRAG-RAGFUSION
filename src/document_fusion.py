def fuse_documents(fusion_documents: list, k=60):
    documents = []
    fused_scores = {}
    
    for docs in fusion_documents:
        for rank, doc in enumerate(docs, start=1):
            if doc.page_content not in fused_scores:
                fused_scores[doc.page_content] = 0
                documents.append(doc)
            fused_scores[doc.page_content] += 1 / (rank + k)
    
    reranked_results = {doc_str: score for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]}
    
    filtered_documents = [doc for doc in documents if doc.page_content in reranked_results]
    return filtered_documents
