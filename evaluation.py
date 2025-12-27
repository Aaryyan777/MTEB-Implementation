import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, cosine_similarity
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, v_measure_score, average_precision_score
from sklearn.cluster import MiniBatchKMeans

class STSEvaluator:
    def __init__(self, sentences1, sentences2, scores):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores

    def __call__(self, model):
        print(f"Encoding {len(self.sentences1)} sentence pairs...")
        embeddings1 = model.encode(self.sentences1)
        embeddings2 = model.encode(self.sentences2)
        
        cosine_dists = paired_cosine_distances(embeddings1, embeddings2)
        cosine_sims = 1 - cosine_dists
        
        spearman_rho, _ = spearmanr(self.scores, cosine_sims)
        return {"spearman_rho": spearman_rho}

class ClassificationEvaluator:
    def __init__(self, train_texts, train_labels, test_texts, test_labels):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

    def __call__(self, model):
        print(f"Encoding {len(self.train_texts)} train and {len(self.test_texts)} test sentences...")
        train_embeddings = model.encode(self.train_texts)
        test_embeddings = model.encode(self.test_texts)
        
        print("Training Logistic Regression...")
        clf = LogisticRegression(max_iter=100, solver='liblinear') 
        clf.fit(train_embeddings, self.train_labels)
        
        predictions = clf.predict(test_embeddings)
        
        acc = accuracy_score(self.test_labels, predictions)
        f1 = f1_score(self.test_labels, predictions, average='macro')
        
        return {"accuracy": acc, "f1": f1}

class ClusteringEvaluator:
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __call__(self, model):
        print(f"Encoding {len(self.sentences)} sentences for clustering...")
        embeddings = model.encode(self.sentences)
        
        k = len(set(self.labels))
        print(f"Running MiniBatchKMeans with k={k}...")
        clustering_model = MiniBatchKMeans(n_clusters=k, batch_size=32, n_init='auto', random_state=42)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        
        v_measure = v_measure_score(self.labels, cluster_assignment)
        return {"v_measure": v_measure}

class PairClassificationEvaluator:
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels # Binary labels (0 or 1)

    def __call__(self, model):
        print(f"Encoding {len(self.sentences1)} pairs...")
        embeddings1 = model.encode(self.sentences1)
        embeddings2 = model.encode(self.sentences2)
        
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        
        # MTEB uses Average Precision based on cosine similarity scores
        ap = average_precision_score(self.labels, cosine_scores)
        return {"average_precision": ap}

class RerankingEvaluator:
    def __init__(self, samples):
        # samples is a list of dicts: {'query': str, 'positive': list[str], 'negative': list[str]}
        self.samples = samples

    def __call__(self, model):
        print(f"Evaluating Reranking on {len(self.samples)} samples...")
        mrr_scores = []
        
        for i, sample in enumerate(self.samples):
            query = sample['query']
            positives = sample['positive']
            negatives = sample['negative']
            
            # Combine candidates: positives first, then negatives
            docs = positives + negatives
            is_relevant = [1] * len(positives) + [0] * len(negatives)
            
            query_emb = model.encode([query])
            docs_emb = model.encode(docs)
            
            scores = cosine_similarity(query_emb, docs_emb)[0]
            
            # Sort by score descending
            ranked_indices = np.argsort(scores)[::-1]
            ranked_relevance = [is_relevant[i] for i in ranked_indices]
            
            # Calculate MRR (Mean Reciprocal Rank)
            try:
                first_relevant_rank = ranked_relevance.index(1) + 1
                mrr_scores.append(1 / first_relevant_rank)
            except ValueError:
                mrr_scores.append(0)
                
            if i > 0 and i % 50 == 0:
                print(f"Processed {i} samples...", end='\r')

        return {"map": np.mean(mrr_scores)} # Using MRR as proxy for MAP in this simple impl

class RetrievalEvaluator:
    def __init__(self, queries, corpus, relevant_docs):
        # queries: dict {qid: text}
        # corpus: dict {docid: text}
        # relevant_docs: dict {qid: set(docids)}
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs

    def __call__(self, model):
        print(f"Encoding {len(self.corpus)} corpus documents...")
        corpus_ids = list(self.corpus.keys())
        corpus_texts = list(self.corpus.values())
        corpus_embeddings = model.encode(corpus_texts)

        print(f"Encoding {len(self.queries)} queries...")
        qids = list(self.queries.keys())
        query_texts = list(self.queries.values())
        query_embeddings = model.encode(query_texts)

        print("Computing similarity...")
        # Full matrix: Queries x Corpus
        scores = cosine_similarity(query_embeddings, corpus_embeddings)

        ndcg_scores = []
        for i, qid in enumerate(qids):
            # Get top 10
            top_k_indices = np.argsort(scores[i])[::-1][:10]
            top_doc_ids = [corpus_ids[idx] for idx in top_k_indices]
            
            # Simple nDCG@10 (binary relevance)
            true_relevant = self.relevant_docs.get(qid, set())
            dcg = 0.0
            idcg = 0.0
            
            for rank, doc_id in enumerate(top_doc_ids):
                if doc_id in true_relevant:
                    dcg += 1.0 / np.log2(rank + 2)
            
            num_rel = len(true_relevant)
            for rank in range(min(num_rel, 10)):
                idcg += 1.0 / np.log2(rank + 2)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)

        return {"ndcg_at_10": np.mean(ndcg_scores)}

class SummarizationEvaluator:
    def __init__(self, machine_summaries, human_summaries, human_scores):
        self.machine_summaries = machine_summaries
        self.human_summaries = human_summaries # List of lists (multiple references)
        self.human_scores = human_scores

    def __call__(self, model):
        print(f"Encoding summaries...")
        machine_embs = model.encode(self.machine_summaries)
        
        # Simple approach: average embedding of human references
        human_embs = []
        for refs in self.human_summaries:
            ref_embs = model.encode(refs)
            human_embs.append(np.mean(ref_embs, axis=0))
        human_embs = np.array(human_embs)
        
        # Ensure 1D arrays
        scores = paired_cosine_distances(machine_embs, human_embs)
        sims = 1 - scores
        
        # Flatten input just in case
        sims = np.array(sims).flatten()
        human_scores = np.array(self.human_scores).flatten()
        
        # Correlation with human quality scores
        spearman_rho, _ = spearmanr(sims, human_scores)
        
        # Ensure scalar
        if hasattr(spearman_rho, 'item'):
             spearman_rho = spearman_rho.item()
             
        return {"spearman_rho": spearman_rho}

class BitextMiningEvaluator:
    def __init__(self, source_sentences, target_sentences, gold_pairs):
        # gold_pairs: list of (src_idx, tgt_idx)
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.gold_pairs = set(gold_pairs)

    def __call__(self, model):
        print(f"Encoding {len(self.source_sentences)} source and {len(self.target_sentences)} target sentences...")
        src_embs = model.encode(self.source_sentences)
        tgt_embs = model.encode(self.target_sentences)
        
        # Find best match for each source in target
        print("Finding matches...")
        scores = cosine_similarity(src_embs, tgt_embs)
        best_matches = np.argmax(scores, axis=1)
        
        # Calculate Precision/Recall/F1
        tp = 0
        for i, match_idx in enumerate(best_matches):
            if (i, match_idx) in self.gold_pairs:
                tp += 1
        
        precision = tp / len(self.source_sentences)
        recall = tp / len(self.gold_pairs) # Assuming 1-to-1 for simplicity here
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"f1": f1}
