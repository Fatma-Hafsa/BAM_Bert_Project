
"""
Projet BAM - Classification multi-tâches avec distillation de connaissances

Ce projet implémente une classification multi-tâches pour prédire :
1. Le sexe de l'auteur (binaire)
2. La période temporelle du texte (10 classes)

Architecture : CamemBERT + distillation de connaissances avec teacher annealing
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertModel, CamembertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import Adam

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from datetime import datetime

# Configuration pour la reproductibilité
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

class Config:
    
    DATA_DIR = '/content/preprocessed_data'
    RESULTS_DIR = '/content/results'
    MODELS_DIR = '/content/models'
    
    BERT_MODEL = 'camembert-base'
    MAX_LENGTH = 256
    
    # Hyperparamètres d'entraînement
    BATCH_SIZE = 8
    NUM_EPOCHS_TEACHER = 4
    NUM_EPOCHS_STUDENT = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # Paramètres de distillation
    TEMPERATURE = 4.0
    ALPHA_DISTILL = 0.7  # Poids de la loss de distillation
    
    # Teacher annealing
    TEACHER_ANNEALING = True
    ANNEALING_START = 0.1
    ANNEALING_END = 0.8
    ANNEALING_MIN_WEIGHT = 0.1
    
    # Architecture
    DROPOUT_RATE = 0.3
    HIDDEN_DIM = 256
    
    # Validation
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    MIN_CHUNKS_PER_BOOK = 3

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Projet BAM - Classification multi-tâches avec distillation")
print(f"Device utilisé: {device}")
print(f"Modèle BERT: {config.BERT_MODEL}")

# =============================================================================
# Chargement et préparation des données
# =============================================================================

def load_data():
    print("\nChargement des données...")
    
    data_files = ['train_chunks.json', 'val_chunks.json', 'test_chunks.json']
    all_data = []
    
    for file_name in data_files:
        file_path = os.path.join(config.DATA_DIR, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"Chargé {file_name}: {len(data)} chunks")
            except Exception as e:
                print(f"Erreur lors du chargement de {file_name}: {e}")
        else:
            print(f"Fichier non trouvé: {file_name}")
    
    if not all_data:
        print("Aucune donnée trouvée, recherche dans le répertoire...")
        # Recherche alternative
        for root, dirs, files in os.walk(config.DATA_DIR):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                sample = data[0]
                                if all(key in sample for key in ['book_id', 'text', 'sexe', 'classe_temporelle']):
                                    all_data.extend(data)
                                    print(f"Trouvé {file}: {len(data)} chunks")
                    except:
                        continue
    
    if not all_data:
        raise ValueError("Aucune donnée valide trouvée")
    
    print(f"Total: {len(all_data)} chunks")
    
    # Statistiques
    gender_dist = Counter([chunk['sexe'] for chunk in all_data])
    temporal_dist = Counter([chunk['classe_temporelle'] for chunk in all_data])
    
    print(f"Distribution par genre: {dict(gender_dist)}")
    print(f"Distribution temporelle: {dict(temporal_dist)}")
    
    return all_data

def split_by_books(data):
    print("\nSplit stratifié par livre...")
    
    # Grouper par livre
    books_by_id = defaultdict(list)
    for chunk in data:
        books_by_id[chunk['book_id']].append(chunk)
    
    # Filtrer les livres avec assez de chunks
    valid_books = {book_id: chunks for book_id, chunks in books_by_id.items()
                   if len(chunks) >= config.MIN_CHUNKS_PER_BOOK}
    
    print(f"Livres valides: {len(valid_books)}/{len(books_by_id)}")
    
    # Métadonnées des livres pour stratification
    book_metadata = []
    for book_id, chunks in valid_books.items():
        first_chunk = chunks[0]
        book_metadata.append({
            'book_id': book_id,
            'num_chunks': len(chunks),
            'author_gender': first_chunk['sexe'],
            'temporal_class': first_chunk['classe_temporelle']
        })
    
    book_df = pd.DataFrame(book_metadata)
    
    # Stratification combinée
    book_df['strat_label'] = (
        book_df['author_gender'].astype(str) + '_' +
        book_df['temporal_class'].astype(str)
    )
    
    # Split train+val vs test
    train_val_books, test_books = train_test_split(
        book_df,
        test_size=config.TEST_SIZE,
        stratify=book_df['strat_label'],
        random_state=RANDOM_SEED
    )
    
    # Split train vs val
    if config.VAL_SIZE > 0:
        val_size_adjusted = config.VAL_SIZE / (1 - config.TEST_SIZE)
        train_books, val_books = train_test_split(
            train_val_books,
            test_size=val_size_adjusted,
            stratify=train_val_books['strat_label'],
            random_state=RANDOM_SEED
        )
    else:
        train_books = train_val_books
        val_books = pd.DataFrame()
    
    # Convertir en chunks
    def books_to_chunks(book_df):
        chunks = []
        for _, book_row in book_df.iterrows():
            chunks.extend(valid_books[book_row['book_id']])
        return chunks
    
    train_chunks = books_to_chunks(train_books)
    val_chunks = books_to_chunks(val_books) if len(val_books) > 0 else []
    test_chunks = books_to_chunks(test_books)
    
    # Vérification de l'absence de leakage
    train_book_ids = set(train_books['book_id'])
    val_book_ids = set(val_books['book_id']) if len(val_books) > 0 else set()
    test_book_ids = set(test_books['book_id'])
    
    assert len(train_book_ids & test_book_ids) == 0, "Data leakage détecté entre train et test!"
    assert len(train_book_ids & val_book_ids) == 0, "Data leakage détecté entre train et val!"
    assert len(val_book_ids & test_book_ids) == 0, "Data leakage détecté entre val et test!"
    
    print(f"Split réalisé sans leakage:")
    print(f"  Train: {len(train_chunks)} chunks de {len(train_books)} livres")
    print(f"  Val:   {len(val_chunks)} chunks de {len(val_books)} livres")
    print(f"  Test:  {len(test_chunks)} chunks de {len(test_books)} livres")
    
    return train_chunks, val_chunks, test_chunks

# =============================================================================
# Dataset et DataLoader
# =============================================================================

class TextDataset(Dataset):
    """Dataset pour les chunks de texte"""
    
    def __init__(self, chunks, tokenizer, max_length=256):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        text = chunk['text']
        
        # Tokenisation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'gender_label': torch.tensor(chunk['sexe'], dtype=torch.long),
            'temporal_label': torch.tensor(chunk['classe_temporelle'], dtype=torch.long),
            'book_id': chunk['book_id']
        }

# =============================================================================
# Architecture du modèle
# =============================================================================

class BertBAM(nn.Module):
    """
    Modèle BERT avec architecture BAM pour classification multi-tâches
    """
    
    def __init__(self, model_name='camembert-base', model_type="unified"):
        super().__init__()
        
        self.model_name = model_name
        self.model_type = model_type  # "unified", "teacher_gender", "teacher_temporal"
        
        # Chargement du modèle BERT
        self.bert = CamembertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Couches de transformation
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        self.shared_projection = nn.Sequential(
            nn.Linear(self.hidden_size, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        # Têtes de classification selon le type de modèle
        if model_type in ["unified", "teacher_gender"]:
            self.gender_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM // 2, 2)  # Binaire: M/F
            )
        
        if model_type in ["unified", "teacher_temporal"]:
            self.temporal_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM // 2, 10)  # 10 périodes temporelles
            )
    
    def forward(self, input_ids, attention_mask):
        # Encodage BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Token [CLS]
        
        # Projection partagée
        shared_features = self.shared_projection(self.dropout(pooled_output))
        
        result = {}
        
        # Classification selon le type de modèle
        if self.model_type in ["unified", "teacher_gender"]:
            gender_logits = self.gender_head(shared_features)
            result["gender_logits"] = gender_logits
        
        if self.model_type in ["unified", "teacher_temporal"]:
            temporal_logits = self.temporal_head(shared_features)
            result["temporal_logits"] = temporal_logits
        
        return result

# =============================================================================
# Fonctions d'entraînement
# =============================================================================

def train_teacher_model(model, train_loader, val_loader, task_type, num_epochs):
    print(f"\nEntraînement du teacher {task_type}...")
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Entraînement
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if task_type == "gender":
                labels = batch['gender_label'].to(device)
                logits_key = "gender_logits"
            else:  # temporal
                labels = batch['temporal_label'].to(device)
                logits_key = "temporal_logits"
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            logits = outputs[logits_key]
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        if val_loader:
            val_acc = evaluate_model(model, val_loader, task_type)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Sauvegarder le meilleur modèle
                os.makedirs(config.MODELS_DIR, exist_ok=True)
                torch.save(model.state_dict(), 
                          os.path.join(config.MODELS_DIR, f"teacher_{task_type}_best.pth"))
        else:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model

def train_student_with_distillation(student_model, teacher_gender, teacher_temporal, 
                                  train_loader, val_loader, num_epochs):
    print("\nEntraînement du student avec distillation...")
    
    optimizer = Adam(student_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        student_model.train()
        teacher_gender.eval()
        teacher_temporal.eval()
        
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gender_labels = batch['gender_label'].to(device)
            temporal_labels = batch['temporal_label'].to(device)
            
            optimizer.zero_grad()
            
            # Prédictions du student
            student_outputs = student_model(input_ids, attention_mask)
            
            # Prédictions des teachers
            with torch.no_grad():
                teacher_gender_outputs = teacher_gender(input_ids, attention_mask)
                teacher_temporal_outputs = teacher_temporal(input_ids, attention_mask)
            
            # Loss de classification standard
            gender_loss = criterion(student_outputs["gender_logits"], gender_labels)
            temporal_loss = criterion(student_outputs["temporal_logits"], temporal_labels)
            classification_loss = (gender_loss + temporal_loss) / 2
            
            # Loss de distillation
            # Teacher annealing: réduction progressive de l'influence des teachers
            if config.TEACHER_ANNEALING:
                progress = (epoch * len(train_loader) + step) / total_steps
                if progress < config.ANNEALING_START:
                    teacher_weight = 1.0
                elif progress > config.ANNEALING_END:
                    teacher_weight = config.ANNEALING_MIN_WEIGHT
                else:
                    # Décroissance linéaire
                    annealing_progress = (progress - config.ANNEALING_START) / (config.ANNEALING_END - config.ANNEALING_START)
                    teacher_weight = 1.0 - annealing_progress * (1.0 - config.ANNEALING_MIN_WEIGHT)
            else:
                teacher_weight = 1.0
            
            # Distillation pour le genre
            student_gender_soft = F.log_softmax(student_outputs["gender_logits"] / config.TEMPERATURE, dim=1)
            teacher_gender_soft = F.softmax(teacher_gender_outputs["gender_logits"] / config.TEMPERATURE, dim=1)
            gender_distill_loss = kl_div(student_gender_soft, teacher_gender_soft) * (config.TEMPERATURE ** 2)
            
            # Distillation pour le temporel
            student_temporal_soft = F.log_softmax(student_outputs["temporal_logits"] / config.TEMPERATURE, dim=1)
            teacher_temporal_soft = F.softmax(teacher_temporal_outputs["temporal_logits"] / config.TEMPERATURE, dim=1)
            temporal_distill_loss = kl_div(student_temporal_soft, teacher_temporal_soft) * (config.TEMPERATURE ** 2)
            
            distillation_loss = (gender_distill_loss + temporal_distill_loss) / 2
            
            # Loss totale
            total_loss_batch = (
                (1 - config.ALPHA_DISTILL) * classification_loss +
                config.ALPHA_DISTILL * teacher_weight * distillation_loss
            )
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        if val_loader:
            val_acc_gender = evaluate_model(student_model, val_loader, "gender")
            val_acc_temporal = evaluate_model(student_model, val_loader, "temporal")
            val_acc_avg = (val_acc_gender + val_acc_temporal) / 2
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            print(f"  Val Acc Gender = {val_acc_gender:.4f}, Temporal = {val_acc_temporal:.4f}, Avg = {val_acc_avg:.4f}")
            print(f"  Teacher weight = {teacher_weight:.3f}")
            
            if val_acc_avg > best_val_acc:
                best_val_acc = val_acc_avg
                torch.save(student_model.state_dict(), 
                          os.path.join(config.MODELS_DIR, "bam_student_best.pth"))
        else:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Teacher weight = {teacher_weight:.3f}")
    
    return student_model

def evaluate_model(model, dataloader, task_type):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if task_type == "gender":
                labels = batch['gender_label'].to(device)
                logits_key = "gender_logits"
            else:  # temporal
                labels = batch['temporal_label'].to(device)
                logits_key = "temporal_logits"
            
            outputs = model(input_ids, attention_mask)
            logits = outputs[logits_key]
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# =============================================================================
# Pipeline principal
# =============================================================================

def main():
    
    # Créer les répertoires
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Charger les données
    data = load_data()
    train_chunks, val_chunks, test_chunks = split_by_books(data)
    
    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(config.BERT_MODEL)
    
    # Datasets
    train_dataset = TextDataset(train_chunks, tokenizer, config.MAX_LENGTH)
    val_dataset = TextDataset(val_chunks, tokenizer, config.MAX_LENGTH) if val_chunks else None
    test_dataset = TextDataset(test_chunks, tokenizer, config.MAX_LENGTH)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    print(f"\nDataLoaders créés:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader) if val_loader else 0} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # 1. Entraîner les teachers spécialisés
    print("\n" + "="*60)
    print("PHASE 1: Entraînement des teachers spécialisés")
    print("="*60)
    
    # Teacher pour le genre
    teacher_gender = BertBAM(config.BERT_MODEL, "teacher_gender").to(device)
    teacher_gender = train_teacher_model(
        teacher_gender, train_loader, val_loader, "gender", config.NUM_EPOCHS_TEACHER
    )
    
    # Teacher pour le temporel
    teacher_temporal = BertBAM(config.BERT_MODEL, "teacher_temporal").to(device)
    teacher_temporal = train_teacher_model(
        teacher_temporal, train_loader, val_loader, "temporal", config.NUM_EPOCHS_TEACHER
    )
    
    # 2. Entraîner le student avec distillation
    print("\n" + "="*60)
    print("PHASE 2: Entraînement du student avec distillation")
    print("="*60)
    
    student_model = BertBAM(config.BERT_MODEL, "unified").to(device)
    student_model = train_student_with_distillation(
        student_model, teacher_gender, teacher_temporal,
        train_loader, val_loader, config.NUM_EPOCHS_STUDENT
    )
    
    # 3. Évaluation finale
    print("\n" + "="*60)
    print("PHASE 3: Évaluation finale")
    print("="*60)
    
    # Charger le meilleur modèle student
    student_model.load_state_dict(
        torch.load(os.path.join(config.MODELS_DIR, "bam_student_best.pth"))
    )
    
    # Évaluation sur le test set
    test_acc_gender = evaluate_model(student_model, test_loader, "gender")
    test_acc_temporal = evaluate_model(student_model, test_loader, "temporal")
    
    print(f"\nRésultats finaux sur le test set:")
    print(f"  Accuracy Genre: {test_acc_gender:.4f}")
    print(f"  Accuracy Temporel: {test_acc_temporal:.4f}")
    print(f"  Accuracy Moyenne: {(test_acc_gender + test_acc_temporal) / 2:.4f}")
    
    # Sauvegarder les résultats
    results = {
        'test_accuracy_gender': test_acc_gender,
        'test_accuracy_temporal': test_acc_temporal,
        'test_accuracy_average': (test_acc_gender + test_acc_temporal) / 2,
        'config': {
            'bert_model': config.BERT_MODEL,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'temperature': config.TEMPERATURE,
            'alpha_distill': config.ALPHA_DISTILL,
            'teacher_annealing': config.TEACHER_ANNEALING
        },
        'data_stats': {
            'train_chunks': len(train_chunks),
            'val_chunks': len(val_chunks),
            'test_chunks': len(test_chunks)
        }
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nRésultats sauvegardés dans {config.RESULTS_DIR}/final_results.json")
    print("Entraînement terminé!")

if __name__ == "__main__":
    main()

