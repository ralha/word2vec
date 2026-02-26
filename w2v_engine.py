import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import random
import numpy as np
import math
import re
from pathlib import Path
import string

import nltk
from nltk.corpus import stopwords
# Ensure the data is there
nltk.download('stopwords')

class Word2VecEngine:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.model = None

    def get_refined_stopwords(self):
        # 1. Start with NLTK defaults
        stop_words = set(stopwords.words('english'))
        
        # 2. Add common missing tokens/conjunctions
        extra_stops = {'at', 'if', 'on', 'the', 'is', 'it', 'to', 'and', 'a'} 
        stop_words.update(extra_stops)
        
        # 3. Add punctuation (often missed if not handled separately)
        stop_words.update(list(string.punctuation))
        
        # 4. Optional: Add "Fairytale" specific noise
        # Since you're prepping fairytales, words like "said" or "upon" 
        # might be so frequent they act like stop words.
        stop_words.update(['said', 'upon', 'would', 'could'])
        
        return stop_words

    def prepare_data(self, corpus_type, min_count, threshold=1e-3):
        corpus_path = f"./{corpus_type}/clean/corpus.txt"
        print(f"--- Loading and Cleaning: {corpus_path} ---")
        raw_sentences = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens: raw_sentences.append(tokens)

        word_counts = Counter([w for s in raw_sentences for w in s])
        total_count = sum(word_counts.values())

        # 1.1 Stopword list (highly recommended for analogies on small data)
        stop_words = self.get_refined_stopwords()
        # 1.2 Subsampling logic: P(w) = 1 - sqrt(threshold / frequency)
        # This keeps rare words and downsamples common ones
        def should_keep(word):
            word_lower = word.lower()

            if word_lower in stop_words: return False # Optional: hard filter
            if word_counts[word_lower] < min_count: return False

            freq = word_counts[word_lower] / total_count
            prob = 1 - math.sqrt(threshold / freq)
            return random.random() > prob
        

        # Subsampling & Min-count filtering
        filtered_corpus = []
        for sentence in raw_sentences:
            filtered_sentence = [w for w in sentence if should_keep(w)]
            if len(filtered_sentence) > 1:
                filtered_corpus.append(filtered_sentence)

        # Rebuild Vocab
        final_counts = Counter([w for s in filtered_corpus for w in s])
        self.word2idx = {w: i for i, w in enumerate(final_counts.keys())}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
        return filtered_corpus, final_counts
    
    def create_negative_table(self,counts, table_size=10**6):
        # Ensure counts are in the correct index order
        sorted_counts = np.array([counts[self.idx2word[i]] for i in range(len(self.word2idx))])        
        # Apply the 3/4 power rule
        pow_counts = np.power(sorted_counts, 0.75)
        probs = pow_counts / pow_counts.sum()
        
        # Pre-generate table for O(1) lookup
        return np.random.choice(len(self.word2idx), size=table_size, p=probs)
    
    def train(self, corpus_type, model, tokenized_corpus, neg_table, window_size=5, epochs=50, batch_size=1024,num_negatives=10):
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 1. Generate Pairs (Skip-gram)
        print("--- Generating Pairs ---")
        pairs = []
        for sentence in tokenized_corpus:
            indices = [self.word2idx[w] for w in sentence]
            for center_idx, center_word in enumerate(indices):
                context_range = range(max(0, center_idx - window_size), 
                                  min(len(indices), center_idx))
                for context_idx in context_range:
                    if context_idx != center_idx:
                        pairs.append((center_word, indices[context_idx]))

        # 2. Training Loop
        table_ptr = 0

        epoch_loss = []
        print("--- Trainning ---")
        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                if len(batch) < batch_size: continue

                centers = torch.LongTensor([c for c, p in batch])
                positives = torch.LongTensor([p for c, p in batch])
                
                # Get negatives from pre-generated table
                if table_ptr + (batch_size * 10) > len(neg_table): 
                    table_ptr = 0
                
                neg_indices = neg_table[table_ptr:table_ptr + (batch_size * num_negatives)]
                negatives = torch.LongTensor(neg_indices).view(batch_size, num_negatives)
                table_ptr += (batch_size * num_negatives)

                optimizer.zero_grad()
                loss = model(centers, positives, negatives)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.6f}, LR: {scheduler.get_last_lr()[0]:.5f}")
            epoch_loss.append(total_loss)
        torch.save(model, f"word2vect_{corpus_type}.pt")
        return epoch_loss
    
    def find_similar_words(self, model, word, top_n=5):
        if word not in self.word2idx:
            raise Exception(f"{word} not in vocabulary")
        
        # Use 'in' embeddings for similarity
        weights = model.in_embed.weight.data
        query_vec = weights[self.word2idx[word]]
        
        # Cosine similarity: (A dot B) / (||A|| * ||B||)
        cos_sim = F.cosine_similarity(query_vec.unsqueeze(0), weights)
        
        scores, indices = torch.topk(cos_sim, top_n + 1)
        print(f"\nTop {top_n} words similar with {word}:")
        for i in range(1, len(indices)): # skip the word itself
            print(f"{self.idx2word[indices[i].item()]}: {scores[i].item():.4f}")

    def find_similar_embedding(self,model,positive, negative, top_n=1):

        for item in positive + negative:
            if item not in self.word2idx:
                raise Exception(f"{item} not in vocabulary")
                
        # Get all embeddings
        weights = model.in_embed.weight.data # (Vocab, Dim)

        # Calculate target vector: King - Man + Woman
        query_vec = weights[self.word2idx[positive[0]]] - weights[self.word2idx[negative[0]]] + weights[self.word2idx[positive[1]]]
        query_vec = query_vec.unsqueeze(0) # (1, Dim)

        cos_sim = F.cosine_similarity(query_vec, weights)
        # Get top results
        scores, indices = torch.topk(cos_sim, top_n+3)
        
        # Filter out the input words from the results
        input_indices = {self.word2idx[w] for w in positive + negative if w in self.word2idx}

        print(f"\nTop {top_n} results for {positive[0]} - {negative[0]} + {positive[1]}:")
        count = 0
        for i in range(len(indices)):
            idx = indices[i].item()
            if idx not in input_indices and count < top_n:
                print(f"{self.idx2word[idx]}: {scores[i].item():.4f}")
                count += 1


# 3. Optimized Model using Dot Product
class Word2VecSGNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        # Initialize weights to small values
        initrange = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_words, pos_words, neg_words):
        # center: (batch), pos: (batch), neg: (batch, num_neg)
        
        v = self.in_embed(center_words).unsqueeze(2)    # (batch, dim, 1)
        u_pos = self.out_embed(pos_words).unsqueeze(1) # (batch, 1, dim)
        u_neg = self.out_embed(neg_words)              # (batch, num_neg, dim)

        # Log-sigmoid loss for positive pairs (Dot product)
        pos_score = torch.bmm(u_pos, v).squeeze()      # (batch)
        pos_loss = F.logsigmoid(pos_score)

        # Log-sigmoid loss for negative pairs
        neg_score = torch.bmm(u_neg, v).squeeze()      # (batch, num_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(1)     # Sum over negative samples

        return -(pos_loss + neg_loss).mean()