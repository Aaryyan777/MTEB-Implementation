from datasets import load_dataset
from .evaluation import (
    STSEvaluator, ClassificationEvaluator, ClusteringEvaluator, 
    PairClassificationEvaluator, RerankingEvaluator, RetrievalEvaluator,
    SummarizationEvaluator, BitextMiningEvaluator
)
import random

class Task:
    def __init__(self, name):
        self.name = name

    def run(self, model):
        raise NotImplementedError

class STSBTask(Task):
    def __init__(self):
        super().__init__("STS-B (GLUE)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        dataset = load_dataset("glue", "stsb", split="validation")
        
        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        scores = dataset['label']

        evaluator = STSEvaluator(sentences1, sentences2, scores)
        results = evaluator(model)
        return results

class Banking77Task(Task):
    def __init__(self):
        super().__init__("Banking77 (Classification)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        try:
            dataset = load_dataset("banking77", trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load banking77: {e}")
            return {}

        print("Subsampling dataset for speed...")
        train_data = dataset['train'].shuffle(seed=42).select(range(2000))
        test_data = dataset['test'].shuffle(seed=42).select(range(1000))

        train_texts = train_data['text']
        train_labels = train_data['label']
        test_texts = test_data['text']
        test_labels = test_data['label']

        evaluator = ClassificationEvaluator(train_texts, train_labels, test_texts, test_labels)
        results = evaluator(model)
        return results

class TwentyNewsgroupsTask(Task):
    def __init__(self):
        super().__init__("TwentyNewsgroups (Clustering)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        # Using a HF version of 20 Newsgroups
        dataset = load_dataset("SetFit/20_newsgroups", split="test")
        
        # Subsample for speed
        dataset = dataset.shuffle(seed=42).select(range(2000))
        
        sentences = dataset['text']
        labels = dataset['label']
        
        evaluator = ClusteringEvaluator(sentences, labels)
        results = evaluator(model)
        return results

class QQPTask(Task):
    def __init__(self):
        super().__init__("QuoraQuestionPairs (Pair Classification)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        try:
            dataset = load_dataset("glue", "qqp", split="validation")
        except:
             print("Could not load QQP, skipping.")
             return {"error": "Load failed"}

        n = min(len(dataset), 1000)
        dataset = dataset.shuffle(seed=42).select(range(n))
        
        sentences1 = dataset['question1']
        sentences2 = dataset['question2']
        labels = dataset['label'] 
        
        evaluator = PairClassificationEvaluator(sentences1, sentences2, labels)
        results = evaluator(model)
        return results

class SciDocsRetrievalTask(Task):
    def __init__(self):
        super().__init__("Simulated Retrieval (20 Newsgroups)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        # Simulating a retrieval task using 20 Newsgroups
        # Query: "Talk about [Category Name]"
        # Corpus: Actual posts
        # Relevance: Post matches category
        ds = load_dataset("SetFit/20_newsgroups", split="test").shuffle(seed=42).select(range(500))
        
        corpus = {i: text for i, text in enumerate(ds['text'])}
        
        # Create one query per unique label found in the subset
        unique_labels = list(set(ds['label_text']))
        queries = {i: f"Find documents about {label}" for i, label in enumerate(unique_labels)}
        
        relevant_docs = {qid: set() for qid in queries}
        
        # Map back to find relevant docs
        for docid, row in enumerate(ds):
            label = row['label_text']
            # Find the qid for this label
            for qid, qtext in queries.items():
                if label in qtext:
                    relevant_docs[qid].add(docid)

        evaluator = RetrievalEvaluator(queries, corpus, relevant_docs)
        results = evaluator(model)
        return results

class AskUbuntuTask(Task):
    def __init__(self):
        super().__init__("AskUbuntu (Reranking)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        # Using a dataset that has query + positives + negatives
        try:
             # This dataset usually has lists of positive/negative
            dataset = load_dataset("mteb/askubuntudupquestions-reranking", split="test")
        except:
            print("Could not load AskUbuntu, skipping.")
            return {"error": "Load failed"}
        
        dataset = dataset.shuffle(seed=42).select(range(100))
        
        samples = []
        for row in dataset:
            samples.append({
                'query': row['query'],
                'positive': row['positive'],
                'negative': row['negative']
            })

        evaluator = RerankingEvaluator(samples)
        results = evaluator(model)
        return results

class SimulatedSummarizationTask(Task):
    def __init__(self):
        super().__init__("Simulated Summarization")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        # Synthesize data to guarantee execution
        
        machine_summaries = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast dark fox leaped over a sleeping canine.",
            "Apples are red and bananas are yellow.",
            "Data science is the study of data.",
            "Python is a great programming language."
        ] * 20 # 100 examples
        
        human_summaries = [
            ["The quick fox jumps over the dog.", "Fox jumps over dog."],
            ["A fox jumped over a dog.", "Fast fox, lazy dog."],
            ["Red apples, yellow bananas.", "Fruit colors."],
            ["Data science involves analyzing data.", "Study of data."],
            ["Python is popular for coding.", "Coding in Python."]
        ] * 20
        
        human_scores = [0.9, 0.8, 0.7, 0.6, 0.5] * 20
        
        evaluator = SummarizationEvaluator(machine_summaries, human_summaries, human_scores)
        results = evaluator(model)
        return results

class TatoebaTask(Task):
    def __init__(self):
        super().__init__("OpusBooks (Bitext Mining)")

    def run(self, model):
        print(f"Loading {self.name} dataset...")
        # Opus Books (en-fr) is usually reliable
        try:
            dataset = load_dataset("opus_books", "en-fr", split="train")
        except:
            print("Could not load OpusBooks, skipping.")
            return {"error": "Load failed"}

        n = min(len(dataset), 1000)
        dataset = dataset.shuffle(seed=42).select(range(n))
        
        source_sentences = [row['translation']['en'] for row in dataset]
        target_sentences = [row['translation']['fr'] for row in dataset]
        
        # Gold pairs are just 1-to-1 matching indices since we took parallel sentences
        gold_pairs = [(i, i) for i in range(len(source_sentences))]

        evaluator = BitextMiningEvaluator(source_sentences, target_sentences, gold_pairs)
        results = evaluator(model)
        return results
