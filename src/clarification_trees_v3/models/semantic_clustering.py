from omegaconf import ListConfig
from abc import ABC, abstractmethod
from typing import List, Tuple
from omegaconf import DictConfig
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor
import asyncio
import functools
from typing import Any

class Clusterer(ABC):
    """
    Abstract base class for clustering text outputs.
    """
    def __init__(self, config: DictConfig, device: str):
        self.config = config
        self.device = device
        # Common config extraction
        self.exemplar_selection_method = config.get("exemplar_selection_method", "random")
        self.executor = ThreadPoolExecutor(max_workers=5)

    @abstractmethod
    def cluster(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        """
        Clusters texts and returns (list of clusters, list of exemplars).
        """
        pass

    async def async_cluster(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        loop = asyncio.get_running_loop()
        func = functools.partial(self.cluster, texts)
        return await loop.run_in_executor(self.executor, func)

    def _select_exemplars(self, clusters: List[List[str]], metadata_clusters: List[List[Any]] = None, embeddings: np.ndarray = None) -> Tuple[List[str], List[Any]]:
        """
        Helper to select the representative 'center' of each cluster based on config.
        """
        exemplars = []
        metadata_exemplars = []
        
        for i, cluster_texts in enumerate(clusters):
            if not cluster_texts:
                continue

            if self.exemplar_selection_method == "closest_to_mean" and embeddings is not None:
                # Need to find indices of these texts in the original embedding list implies 
                # we need to track indices. This helper assumes we can calculate fresh or 
                # provided embeddings.
                # For simplicity in this helper, we'll assume the caller handles complex 
                # embedding math (like in SemanticClusterer) or we fallback to length/random.
                pass 
            
            # Fallback methods that don't strictly require embeddings
            if self.exemplar_selection_method == "longest":
                exemplar_idx = np.argmax([len(t) for t in cluster_texts])
                exemplars.append(cluster_texts[exemplar_idx])
                if metadata_clusters is not None:
                    metadata_exemplars.append(metadata_clusters[i][exemplar_idx])
                else:
                    metadata_exemplars.append(None)
            elif self.exemplar_selection_method == "shortest":
                exemplar_idx = np.argmin([len(t) for t in cluster_texts])
                exemplars.append(cluster_texts[exemplar_idx])
                if metadata_clusters is not None:
                    metadata_exemplars.append(metadata_clusters[i][exemplar_idx])
                else:
                    metadata_exemplars.append(None)
            elif self.exemplar_selection_method == "random" or self.exemplar_selection_method == "first":
                exemplar_idx = 0
                exemplars.append(cluster_texts[exemplar_idx])
                if metadata_clusters is not None:
                    metadata_exemplars.append(metadata_clusters[i][exemplar_idx])
                else:
                    metadata_exemplars.append(None)
            else:
                # Default fallback
                exemplars.append(cluster_texts[0])
                if metadata_clusters is not None:
                    metadata_exemplars.append(metadata_clusters[i][0])
                else:
                    metadata_exemplars.append(None)
                
        return exemplars, metadata_exemplars


class SemanticClusterer(Clusterer):
    def __init__(self, config: DictConfig, device: str):
        super().__init__(config, device)
        self.model_name = config.sentence_transformers_key
        self.similarity_threshold = config.similarity_threshold
        self.clustering_method = config.clustering_method
        
        # Initialize Agglomerative Clustering if selected
        if self.clustering_method == "agglomerative":
            dist_threshold = 1.0 - self.similarity_threshold
            self.clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_threshold,
                metric='precomputed',
                linkage='complete'
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        print(f"Loading Embedding Model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def cluster(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        if not texts:
            return [], []

        # 1. Encode
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # 2. Compute Distance Matrix
        cosine_sim_matrix = np.inner(embeddings, embeddings)
        cosine_dist_matrix = 1.0 - cosine_sim_matrix
        cosine_dist_matrix = np.maximum(cosine_dist_matrix, 0.0)

        # 3. Cluster
        cluster_labels = self.clusterer.fit_predict(cosine_dist_matrix)
        
        # 4. Group texts by label
        n_clusters = len(set(cluster_labels))
        grouped_clusters = [[] for _ in range(n_clusters)]
        grouped_indices = [[] for _ in range(n_clusters)]
        
        # AgglomerativeClustering labels might not be contiguous 0..N, so we map them
        unique_labels = sorted(list(set(cluster_labels)))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        for text_idx, label in enumerate(cluster_labels):
            mapped_idx = label_to_idx[label]
            grouped_clusters[mapped_idx].append(texts[text_idx])
            grouped_indices[mapped_idx].append(text_idx)

        # 5. Select Exemplars
        semantic_centers = []
        for cluster_idx, indices in enumerate(grouped_indices):
            cluster_texts = grouped_clusters[cluster_idx]
            
            if self.exemplar_selection_method == "closest_to_mean":
                if len(indices) == 1:
                    best_idx = 0
                else:
                    cluster_embeddings = embeddings[indices]
                    mean_embedding = np.mean(cluster_embeddings, axis=0)
                    distances = np.linalg.norm(cluster_embeddings - mean_embedding, axis=1)
                    best_idx = np.argmin(distances)
                semantic_centers.append(cluster_texts[best_idx])
            else:
                # Fallback to generic selection methods
                # Re-using the logic from parent, but applying it locally
                if self.exemplar_selection_method == "longest":
                    semantic_centers.append(max(cluster_texts, key=len))
                elif self.exemplar_selection_method == "shortest":
                    semantic_centers.append(min(cluster_texts, key=len))
                else:
                    semantic_centers.append(cluster_texts[0])

        return grouped_clusters, semantic_centers


class SlowBidirectionalEntailmentClusterer(Clusterer):
    def __init__(self, config: DictConfig, device: str):
        super().__init__(config, device)
        self.model_name = config.cross_encoder_key
        self.similarity_threshold = config.entailment_threshold
        
        print(f"Loading Cross-Encoder: {self.model_name}...")
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        # Map labels. Most NLI models (like deberta-v3) use:
        # 0: Contradiction, 1: Entailment, 2: Neutral
        # We verify this via config if possible, but standard DeBERTa-v3-NLI follows this.
        self.entailment_label_index = 1

    def _check_bidirectional_entailment(self, text_a: str, text_b: str) -> bool:
        """
        Checks if A -> B AND B -> A.
        """
        # TODO: Add support for doing N^2 entailment checks in parallel
        inputs = [[text_a, text_b], [text_b, text_a]]
        scores = self.model.predict(inputs) # Returns logits [batch, 3]
        # Convert to probabilities
        scores = torch.softmax(torch.tensor(scores), dim=1).numpy()

        a_entails_b = scores[0][self.entailment_label_index] >= self.similarity_threshold
        b_entails_a = scores[1][self.entailment_label_index] >= self.similarity_threshold

        # print(f"Checking bidirectional entailment for: {text_a} and {text_b}")
        # print(f"A entails B: {a_entails_b} ({scores[0][self.entailment_label_index]})")
        # print(f"B entails A: {b_entails_a} ({scores[1][self.entailment_label_index]})")
        
        return a_entails_b and b_entails_a
        
        # # Argmax to get label
        # preds = np.argmax(scores, axis=1)
        
        # # Check if both are entailment (index 1)
        # return (preds[0] == self.entailment_label_index) and (preds[1] == self.entailment_label_index)

    def cluster(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        if not texts:
            return [], []

        # We use a Greedy Clustering algorithm for Entailment.
        # It is O(N*K) where N is num_texts and K is num_clusters.
        
        clusters = [] # List of lists of strings
        
        # We keep track of the "representative" (first item) of each cluster 
        # to minimize comparisons.
        cluster_representatives = [] 

        print(f"Clustering {len(texts)} texts using Bidirectional Entailment...")
        
        for text in texts:
            placed = False
            
            # 1. Try to fit into existing clusters
            for i, rep in enumerate(cluster_representatives):
                # Compare new text against the cluster representative
                if self._check_bidirectional_entailment(text, rep):
                    clusters[i].append(text)
                    placed = True
                    break
            
            # 2. If no fit, create new cluster
            if not placed:
                clusters.append([text])
                cluster_representatives.append(text)

        # 3. Select Exemplars
        # Since we don't have embeddings, "closest_to_mean" is invalid.
        # We fallback to simple heuristics.
        exemplars, metadata_exemplars = self._select_exemplars(clusters)
        
        return clusters, exemplars

class BidirectionalEntailmentClusterer(Clusterer):
    def __init__(self, config: DictConfig, device: str):
        super().__init__(config, device)
        self.model_name = config.cross_encoder_key
        self.similarity_threshold = config.entailment_threshold
        
        print(f"Loading Cross-Encoder: {self.model_name}...")
        if isinstance(self.device, list) or isinstance(self.device, ListConfig):
            assert len(self.device) == 1, "Cross-encoder only supports single GPU"
            self.device_str = f"cuda:{self.device[0]}"
        elif str(self.device).isnumeric():
            self.device_str = f"cuda:{self.device}"
        elif isinstance(self.device, str):
            self.device_str = self.device
        else:
            raise ValueError(f"Invalid device: {self.device} ({type(self.device)})")
        print(f"Using device: {self.device_str}")
        self.model = CrossEncoder(self.model_name, device=self.device_str)
        self.entailment_label_index = 1

    def compute_entailments(self, statements: list[tuple[str, str]]):
        """
        Computes the entailment probabilities of each statement.

        Args:
            statements: A list of tuples of strings, where each tuple is a pair of statements that we will compare as A->B

        Returns:
            A list of floats, where each float is the entailment probability of the corresponding statement.
        """
        if not statements:
            return []
            
        entailment_scores = self.model.predict(
            statements,
            batch_size=512,
            show_progress_bar=False
        )
        probs = torch.softmax(torch.tensor(entailment_scores), dim=1).numpy()
        entailment_scores = probs[:, self.entailment_label_index]
        return entailment_scores

    async def async_compute_entailments(self, statements: list[tuple[str, str]]):
        loop = asyncio.get_running_loop()
        func = functools.partial(self.compute_entailments, statements)
        return await loop.run_in_executor(self.executor, func)

    def compute_biconditional_entailments(self, statements: list[tuple[str, str]]):
        """
        Computes the biconditional entailments of each statement as the minimum of the
        entailment probabilities of the statement and its reverse.

        Args:
            statements: A list of tuples of strings, where each tuple is a pair of statements.

        Returns:
            A list of floats, where each float is the biconditional entailment of the corresponding statement.
        """
        if not statements:
            return []
            
        inputs = []
        for i in range(len(statements)):
            inputs.append([statements[i][0], statements[i][1]])
            inputs.append([statements[i][1], statements[i][0]])

        scores_logits = self.model.predict(
            inputs, 
            batch_size=512,
            show_progress_bar=False
        )

        probs = torch.softmax(torch.tensor(scores_logits), dim=1).numpy()
        entailment_scores = probs[:, self.entailment_label_index]

        biconditional_entailments = []
        for i in range(len(statements)):
            biconditional_entailments.append(min(entailment_scores[2*i], entailment_scores[2*i + 1]))
        
        return biconditional_entailments

    async def async_compute_biconditional_entailments(self, statements: list[tuple[str, str]]):
        loop = asyncio.get_running_loop()
        func = functools.partial(self.compute_biconditional_entailments, statements)
        return await loop.run_in_executor(self.executor, func)

    def cluster(self, texts: List[str], metadata: list[Any] | None = None) -> Tuple[List[List[str]], List[str], List[List[Any]], List[Any]]:
        if not texts:
            return [], [], [], []
        if metadata is None:
            metadata = [None] * len(texts)
            
        n = len(texts)
        if n == 1:
            return [texts], texts, [metadata], metadata

        # 1. Prepare N^2 inputs (All permutations)
        # We need (A, B) and (B, A) for all pairs to compute the full matrix.
        # For N=20, this is 400 inference pairs.
        inputs = []
        for i in range(n):
            for j in range(n):
                inputs.append([texts[i], texts[j]])

        # 2. Batched Inference
        # Returns logits [batch_size, num_labels]
        scores_logits = self.model.predict(
            inputs, 
            batch_size=512,
            show_progress_bar=False
        )

        # 3. Convert to Entailment Probabilities
        # Softmax over the label dimension (usually 3: Contradiction, Entailment, Neutral)
        probs = torch.softmax(torch.tensor(scores_logits), dim=1).numpy()
        entailment_scores = probs[:, self.entailment_label_index]

        # 4. Reshape into N x N matrix
        # matrix[i][j] = Score(Text_i -> Text_j)
        score_matrix = entailment_scores.reshape((n, n))

        # 5. Calculate Bidirectional Similarity
        # Sim(A, B) = min( Score(A->B), Score(B->A) )
        # We symmetrize the matrix by taking the element-wise minimum of M and M.T
        similarity_matrix = np.minimum(score_matrix, score_matrix.T)

        # Ensure diagonal is perfect (Self-entailment is always 1.0)
        np.fill_diagonal(similarity_matrix, 1.0)

        # 6. Clustering
        # We use Agglomerative Clustering with 'precomputed' affinity.
        # linkage='complete' ensures ALL members of a cluster are close to each other 
        # (similar to your strict entailment check), not just close to a "center".
        # We convert similarity to distance: distance = 1 - similarity
        distance_matrix = 1 - similarity_matrix
        
        # We use a distance threshold derived from your similarity threshold
        dist_threshold = 1 - self.similarity_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete', 
            distance_threshold=dist_threshold
        )
        
        labels = clustering.fit_predict(distance_matrix)

        # 7. Group Results
        clusters: List[List[str]] = [[] for _ in range(max(labels) + 1)]
        metadata_clusters: List[List[Any]] = [[] for _ in range(max(labels) + 1)]
        for text_idx, cluster_id in enumerate(labels):
            clusters[cluster_id].append(texts[text_idx])
            metadata_clusters[cluster_id].append(metadata[text_idx])

        # 8. Select Exemplars (e.g., shortest text, or central-most)
        exemplars, metadata_exemplars = self._select_exemplars(clusters, metadata_clusters)

        return clusters, exemplars, metadata_clusters, metadata_exemplars

    async def async_cluster(self, texts: List[str], metadata: list[Any] | None = None) -> Tuple[List[List[str]], List[str], List[List[Any]], List[Any]]:
        loop = asyncio.get_running_loop()
        func = functools.partial(self.cluster, texts, metadata)
        return await loop.run_in_executor(self.executor, func)

class HybridClusterer(Clusterer):
    """
    Two-stage clustering:
    1. Filter candidates using Cosine Similarity (Bi-Encoder).
    2. Verify logic using Biconditional Entailment (Cross-Encoder).
    
    This is faster than pure NLI and fixes 'hallucinated' entailment on disjoint topics.
    """
    def __init__(self, config: DictConfig, device: str):
        super().__init__(config, device)
        
        # 1. Bi-Encoder for Embeddings (Topic Similarity)
        self.embedding_model_name = config.sentence_transformers_key
        print(f"Loading Bi-Encoder (Filter): {self.embedding_model_name}...")
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        
        # 2. Cross-Encoder for NLI (Logical Equivalence)
        self.nli_model_name = config.cross_encoder_key
        print(f"Loading Cross-Encoder (Verifier): {self.nli_model_name}...")
        self.nli_model = CrossEncoder(self.nli_model_name, device=self.device)

        # Thresholds
        # 0.65-0.75 is usually good for "same topic"
        self.sim_threshold = config.get("similarity_threshold", 0.70) 
        # 0.5 is standard for CrossEncoders, but 0.7 is safer
        self.nli_threshold = config.get("entailment_threshold", 0.5)
        
        # Auto-detect entailment index (usually 1)
        self.entailment_label_index = 1

    def _check_bidirectional_entailment_batch(self, candidate_text: str, center_text: str) -> bool:
        """
        Checks A <-> B.
        """
        inputs = [[candidate_text, center_text], [center_text, candidate_text]]
        
        # Predict returns logits. We don't necessarily need full softmax if we just check raw scores,
        # but consistency with threshold is easier with probabilities.
        scores = self.nli_model.predict(inputs, show_progress_bar=False)
        probs = torch.softmax(torch.tensor(scores), dim=1).numpy()
        
        # Check A -> B
        a_to_b = probs[0][self.entailment_label_index]
        if a_to_b < self.nli_threshold:
            return False # Fail fast
            
        # Check B -> A
        b_to_a = probs[1][self.entailment_label_index]
        return b_to_a >= self.nli_threshold

    def cluster(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        if not texts:
            return [], []

        # 1. Pre-compute embeddings for all texts (Fast, Matrix operation)
        # This is O(N) relative to the expensive NLI check
        print(f"Embedding {len(texts)} texts for pre-filtering...")
        all_embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        clusters: List[List[str]] = []
        cluster_centers_indices: List[int] = [] # Indices of the representatives in the original list
        
        # To support "closest_to_mean" later, we need to map final clusters to embeddings.
        # But since we use a Greedy approach, the "center" implies the first element.
        
        for i, text in enumerate(texts):
            current_embedding = all_embeddings[i]
            
            # If no clusters exist, create the first one
            if not clusters:
                clusters.append([text])
                cluster_centers_indices.append(i)
                continue

            # 2. Coarse Filter: Cosine Similarity
            # Compare current text against ALL existing cluster centers at once
            center_embeddings = all_embeddings[cluster_centers_indices]
            
            # util.cos_sim returns [1, num_centers]
            cos_scores = util.cos_sim(current_embedding, center_embeddings)[0]
            
            # Find indices of centers that are topically similar enough
            # sort_descending=True ensures we check the most similar center first
            candidate_indices = torch.where(cos_scores >= self.sim_threshold)[0]
            
            # Optimization: If we have candidates, sort them by similarity score.
            # This makes it more likely to hit the correct cluster on the first NLI check.
            if len(candidate_indices) > 0:
                # Get values and indices, then sort indices by values
                candidate_scores = cos_scores[candidate_indices]
                # torch.argsort is ascending, so we reverse
                sorted_order = torch.argsort(candidate_scores, descending=True)
                sorted_candidate_indices = candidate_indices[sorted_order].tolist()
            else:
                sorted_candidate_indices = []

            match_found = False
            
            # 3. Fine Verification: NLI
            for center_idx_pointer in sorted_candidate_indices:
                # Map back to the actual cluster index
                # center_idx_pointer is the index in the `cluster_centers_indices` list
                real_center_text = texts[cluster_centers_indices[center_idx_pointer]]
                
                # Run expensive check
                if self._check_bidirectional_entailment_batch(text, real_center_text):
                    clusters[center_idx_pointer].append(text)
                    match_found = True
                    break
            
            if not match_found:
                clusters.append([text])
                cluster_centers_indices.append(i)

        # 4. Exemplar Selection
        # Since we calculated embeddings at the start, we can pass them to the helper
        # to support "closest_to_mean" selection if configured.
        # Note: The helper expects numpy array, we have tensor.
        embeddings_np = all_embeddings.cpu().numpy()
        exemplars, metadata_exemplars = self._select_exemplars(clusters, embeddings=embeddings_np)
        
        return clusters, exemplars

# --- Execution Block ---
if __name__ == "__main__":
    # Mocking Omegaconf for standalone execution
    import sys
    import dotenv
    dotenv.load_dotenv()
    
    # Define configurations
    semantic_cfg = DictConfig({
        "sentence_transformers_key": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.85, # 0.85 is usually the "same meaning" sweet spot
        "clustering_method": "agglomerative",
        "exemplar_selection_method": "closest_to_mean"
    })

    entailment_cfg = DictConfig({
        "cross_encoder_key": "cross-encoder/nli-deberta-v3-base", 
        "exemplar_selection_method": "shortest", # Logic-based clustering often prefers concise answers
        "entailment_threshold": 0.5
    })

    hybrid_cfg = DictConfig({
        "sentence_transformers_key": "all-MiniLM-L6-v2",
        "cross_encoder_key": "cross-encoder/nli-deberta-v3-base",
        "similarity_threshold": 0.30,  # Tune this for "strictness"
        "entailment_threshold": 0.5,
        "exemplar_selection_method": "shortest"
    })

    test_texts = [
        # --- Group 1: Simple Paraphrasing (Both methods should group these) ---
        "The sky is blue.",
        "Blue is the color of the sky.",
        "The color of the sky is blue.",
        "Looking up, I see a blue sky.",

        # --- Group 2: The "Lexical Overlap" Trap (Embeddings often fail here, Entailment succeeds) ---
        "The ocean is blue.",                    # High overlap with "sky is blue", but different fact
        "The sky is not blue.",                  # Negation: Embeddings cluster this with "sky is blue", Entailment separates
        "The sky is green.",                     # Contradiction

        # --- Group 3: Math & Logic (Entailment shines here) ---
        "x + y = 10",
        "y = 10 - x",                            # Logically equivalent to x + y = 10
        "x = 10 - y",                            # Logically equivalent
        "y = x - 10",                            # DIFFERENT equation (y - x = -10)
        "x + y = 12",                            # Small number change, huge semantic difference

        # --- Group 4: Code Syntax (Python) ---
        "print('Hello World')",
        "print(\"Hello World\")",                # Identical code
        "sys.stdout.write('Hello World\\n')",    # Functionally equivalent, lexically distinct
        "print('Hello Python')",                 # Different output

        # --- Group 5: Conversational Refusals (Common in RLHF/Safety) ---
        "I cannot answer that question.",
        "I'm sorry, but I can't provide that information.",
        "As an AI language model, I am unable to assist with this request.",
        "I can answer that question.",           # Opposite meaning, high lexical overlap

        # --- Group 6: Sentiment & Nuance ---
        "The movie was good.",
        "The film was excellent.",               # "Excellent" implies "good" (entailment), but "good" doesn't strictly imply "excellent". 
                                                 # (Bidirectional entailment might separate these depending on strictness)
        "The movie was bad.",                    # Opposite sentiment
        "The movie was not bad.",                # Litotes (often means "average" or "good")

        # --- Group 7: Entity Resolution ---
        "Barack Obama was the 44th president of the USA.",
        "The 44th US president was Obama.",
        "Obama served as the 44th president.",
        "Donald Trump was the 45th president.",  # Different entity, same structure

        # --- Group 8: GSM8K Style Reasoning (Word Problems) ---
        "Sally has 3 apples and eats 1. She has 2 left.",
        "Sally started with 3 apples, ate one, and now implies she has 2.",
        "Sally has 2 apples.",                   # This is the *conclusion* of the previous two, but misses the premise. 
                                                 # Entailment might group if checking "Does A imply B?", but strictly they are different statements.
        
        # --- Group 9: Hallucinations / Factual Errors ---
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower is in Berlin.",        # Hallucination
        "Paris is the home of the Eiffel Tower."

        # --- Group 10: Subject vs. Object (The "Bag of Words" Trap) ---
        "Who called John?",                      # John is the receiver
        "Who did John call?",                    # John is the caller
        "Whom was called by John?",              # Passive voice, same as "Who did John call?"
        "John called who?",                      # Informal phrasing of "Who did John call?"

        # --- Group 11: Modals of Obligation vs. Possibility ---
        "Can I reset my password?",              # Asking about ability/possibility
        "May I reset my password?",              # Asking for permission (often semantically close to 'Can')
        "Must I reset my password?",             # Asking about obligation (Very different meaning)
        "Should I reset my password?",           # Asking for advice (Different meaning)

        # --- Group 12: Temporal Nuance (Tense shifting) ---
        "When is the train leaving?",            # Future intent
        "When did the train leave?",             # Past fact
        "What time does the train depart?",      # Paraphrase of "When is the train leaving?"
        "Has the train left?",                   # Yes/No verification of past event

        # --- Group 13: Pragmatic Equivalence (Different words, Same Intent) ---
        "How much does this cost?",
        "What is the price of this item?",       # Semantically identical to above
        "Is this item expensive?",               # Related topic, but distinct question (Subjective vs Objective)
        "Can I afford this?",                    # Distinct question (dependent on user's wallet, not just price)

        # --- Group 14: Scope and Specificity ---
        "Where is the nearest bank?",
        "Where is the nearest Citi bank?",       # Specific entity (subset of above)
        "Is there a bank near here?",            # Yes/No existence check vs Location retrieval
        "How do I get to the bank?",             # Navigation instructions vs Location coordinates
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    # 1. Run Semantic Clustering (Embedding Based)
    print("--- Running Semantic Clusterer (Embeddings) ---")
    sem_clusterer = SemanticClusterer(semantic_cfg, device)
    s_clusters, s_exemplars = sem_clusterer.cluster(test_texts)
    
    for i, (cluster, center) in enumerate(zip(s_clusters, s_exemplars)):
        print(f"Cluster {i+1} [Center: '{center}']: {cluster}")

    print("\n" + "="*50 + "\n")

    # 2. Run Entailment Clustering (Logic Based)
    print("--- Running Entailment Clusterer (Logic/NLI) ---")
    # Note: Entailment is slower but stricter. 
    # It should correctly separate "The sky is blue" from "The ocean is blue" 
    # even if embeddings think they are similar.
    ent_clusterer = BidirectionalEntailmentClusterer(entailment_cfg, device)
    e_clusters, e_exemplars, _, _ = ent_clusterer.cluster(test_texts)
    
    for i, (cluster, center) in enumerate(zip(e_clusters, e_exemplars)):
        print(f"Cluster {i+1} [Center: '{center}']: {cluster}")


    print("\n" + "="*50 + "\n")

    # 3. Run Hybrid Clustering (Embeddings + NLI)
    print("--- Running Hybrid Clusterer (Embeddings + NLI) ---")
    hybrid_clusterer = HybridClusterer(hybrid_cfg, device)
    h_clusters, h_exemplars = hybrid_clusterer.cluster(test_texts)
    
    for i, (cluster, center) in enumerate(zip(h_clusters, h_exemplars)):
        print(f"Cluster {i+1} [Center: '{center}']: {cluster}")
