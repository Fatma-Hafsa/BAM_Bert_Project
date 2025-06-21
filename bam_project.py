"""
Projet BAM - Classification multi-tâches avec distillation de connaissances
Architecture : CamemBERT + distillation avec teacher annealing + feature annealing
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertModel, CamembertTokenizer, FlaubertModel, FlaubertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from datetime import datetime

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
    
    BATCH_SIZE = 8
    NUM_EPOCHS_TEACHER = 4
    NUM_EPOCHS_STUDENT = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    TEMPERATURE = 4.0
    ALPHA_DISTILL = 0.7
    
    TEACHER_ANNEALING = True
    ANNEALING_START = 0.1
    ANNEALING_END = 0.8
    ANNEALING_MIN_WEIGHT = 0.1
    
    FEATURE_ANNEALING = True
    ATTENTION_ANNEALING_START = 0.2
    ATTENTION_ANNEALING_END = 0.9
    
    DROPOUT_RATE = 0.3
    HIDDEN_DIM = 256
    
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    MIN_CHUNKS_PER_BOOK = 3
    N_BOOTSTRAP = 1000
    CONFIDENCE_LEVEL = 0.95
    
    # Constantes pour évaluation et interface
    TEMPORAL_PERIODS = {
        0: "1820-1839", 1: "1840-1859", 2: "1860-1879", 3: "1880-1899", 4: "1900-1919",
        5: "1920-1939", 6: "1940-1959", 7: "1960-1979", 8: "1980-1999", 9: "2000-2020"
    }
    
    TEMPORAL_MIDPOINTS = {
        0: 1830, 1: 1850, 2: 1870, 3: 1890, 4: 1910,
        5: 1930, 6: 1950, 7: 1970, 8: 1990, 9: 2010
    }

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Projet BAM - Classification multi-tâches avec distillation")
print("Architecture:", config.BERT_MODEL)
print("Device:", device)
print("Distillation: alpha={}, temperature={}".format(config.ALPHA_DISTILL, config.TEMPERATURE))

def load_data():
    print("\nChargement des données")
    
    data_files = ['train_chunks.json', 'val_chunks.json', 'test_chunks.json']
    all_data = []
    
    for file_name in data_files:
        file_path = os.path.join(config.DATA_DIR, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print("Chargé {}: {} chunks".format(file_name, len(data)))
            except Exception as e:
                print("Erreur {}: {}".format(file_name, e))
        else:
            print("Fichier non trouvé: {}".format(file_name))
    
    if not all_data:
        print("\nRecherche d'autres formats de données...")
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
                                    print("Trouvé {}: {} chunks".format(file, len(data)))
                    except:
                        continue
    
    if not all_data:
        raise ValueError("Aucune donnée trouvée")
    
    print("Analyse du corpus: {} chunks".format(len(all_data)))
    
    books_by_id = defaultdict(list)
    for chunk in all_data:
        books_by_id[chunk['book_id']].append(chunk)
    
    valid_books = {book_id: chunks for book_id, chunks in books_by_id.items()
                   if len(chunks) >= config.MIN_CHUNKS_PER_BOOK}
    
    print("Livres avec >= {} chunks: {}/{}".format(config.MIN_CHUNKS_PER_BOOK, len(valid_books), len(books_by_id)))
    
    total_books = len(valid_books)
    total_chunks = len(all_data)
    
    print("Corpus complet: {} livres, {} chunks".format(total_books, total_chunks))
    
    if total_chunks > 200000:
        target_books_per_class = 50
        print("Très gros corpus détecté, limitation à {} livres/classe".format(target_books_per_class))
    elif total_chunks > 100000:
        target_books_per_class = 80
        print("Gros corpus détecté, limitation à {} livres/classe".format(target_books_per_class))
    elif total_chunks > 50000:
        target_books_per_class = 100
        print("Corpus moyen détecté, limitation à {} livres/classe".format(target_books_per_class))
    else:
        target_books_per_class = None
        print("Petit corpus: utilisation de tous les {} livres".format(total_books))
    
    if target_books_per_class is not None:
        selected_chunks = []
        books_by_temporal_class = defaultdict(list)
        
        for book_id, chunks in valid_books.items():
            temporal_class = chunks[0]['classe_temporelle']
            books_by_temporal_class[temporal_class].append((book_id, chunks))
        
        total_selected_books = 0
        for temporal_class, books_list in books_by_temporal_class.items():
            random.shuffle(books_list)
            selected_books = books_list[:target_books_per_class]
            
            for book_id, chunks in selected_books:
                selected_chunks.extend(chunks)
            
            total_selected_books += len(selected_books)
            print("Classe {}: {}/{} livres sélectionnés".format(temporal_class, len(selected_books), len(books_list)))
        
        all_data = selected_chunks
        print("Échantillon final: {} livres, {} chunks".format(total_selected_books, len(all_data)))
    
    gender_dist = Counter([chunk['sexe'] for chunk in all_data])
    temporal_dist = Counter([chunk['classe_temporelle'] for chunk in all_data])
    
    print("\nDistribution finale:")
    print("Genre:", dict(gender_dist))
    print("Temporel:", dict(temporal_dist))
    print("Livres finaux:", len(set(chunk['book_id'] for chunk in all_data)))
    
    return all_data

def split_by_books(data):
    print("\nSplit stratifié par livre")
    
    books_by_id = defaultdict(list)
    for chunk in data:
        books_by_id[chunk['book_id']].append(chunk)
    
    valid_books = {book_id: chunks for book_id, chunks in books_by_id.items()
                   if len(chunks) >= config.MIN_CHUNKS_PER_BOOK}
    
    print("Livres valides: {}/{}".format(len(valid_books), len(books_by_id)))
    
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
    
    book_df['strat_label'] = (
        book_df['author_gender'].astype(str) + '_' +
        book_df['temporal_class'].astype(str)
    )
    
    strata_counts = book_df['strat_label'].value_counts()
    min_strata = strata_counts.min()
    
    if min_strata < 3:
        print("Ajustement split: min strata = {}".format(min_strata))
        adj_test_size = min(config.TEST_SIZE, 1/min_strata)
        adj_val_size = min(config.VAL_SIZE, 1/min_strata)
    else:
        adj_test_size = config.TEST_SIZE
        adj_val_size = config.VAL_SIZE
    
    train_val_books, test_books = train_test_split(
        book_df,
        test_size=adj_test_size,
        stratify=book_df['strat_label'],
        random_state=RANDOM_SEED
    )
    
    if adj_val_size > 0:
        try:
            adj_val_for_trainval = adj_val_size / (1 - adj_test_size)
            train_books, val_books = train_test_split(
                train_val_books,
                test_size=adj_val_for_trainval,
                stratify=train_val_books['strat_label'],
                random_state=RANDOM_SEED
            )
        except ValueError:
            print("Split validation impossible")
            train_books = train_val_books
            val_books = pd.DataFrame()
    else:
        train_books = train_val_books
        val_books = pd.DataFrame()
    
    def books_to_chunks(book_df):
        chunks = []
        for _, book_row in book_df.iterrows():
            chunks.extend(valid_books[book_row['book_id']])
        return chunks
    
    train_chunks = books_to_chunks(train_books)
    val_chunks = books_to_chunks(val_books) if len(val_books) > 0 else []
    test_chunks = books_to_chunks(test_books)
    
    train_book_ids = set(train_books['book_id'])
    val_book_ids = set(val_books['book_id']) if len(val_books) > 0 else set()
    test_book_ids = set(test_books['book_id'])
    
    assert len(train_book_ids & test_book_ids) == 0, "LEAKAGE train-test détecté!"
    assert len(train_book_ids & val_book_ids) == 0, "LEAKAGE train-val détecté!"
    assert len(val_book_ids & test_book_ids) == 0, "LEAKAGE val-test détecté!"
    
    print("Split sans leakage:")
    print("Train: {} chunks de {} livres".format(len(train_chunks), len(train_books)))
    print("Val: {} chunks de {} livres".format(len(val_chunks), len(val_books)))
    print("Test: {} chunks de {} livres".format(len(test_chunks), len(test_books)))
    
    return train_chunks, val_chunks, test_chunks

class TextDataset(Dataset):
    def __init__(self, chunks, tokenizer, max_length=256):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("Dataset: {} chunks".format(len(chunks)))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        text = chunk['text']
        
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

class BertBAM(nn.Module):
    def __init__(self, model_name='camembert-base', model_type="unified"):
        super().__init__()
        
        self.model_name = model_name
        self.model_type = model_type
        
        try:
            if 'camembert' in model_name:
                self.bert = CamembertModel.from_pretrained(model_name)
            elif 'flaubert' in model_name:
                self.bert = FlaubertModel.from_pretrained(model_name)
            else:
                self.bert = CamembertModel.from_pretrained('camembert-base')
            print("BERT chargé: {} ({})".format(model_name, model_type))
        except Exception as e:
            print("Erreur BERT: {}".format(e))
            self.bert = CamembertModel.from_pretrained('camembert-base')
        
        self.hidden_size = self.bert.config.hidden_size
        
        if config.FEATURE_ANNEALING:
            self.feature_attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        self.shared_projection = nn.Sequential(
            nn.Linear(self.hidden_size, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        if model_type in ["unified", "teacher_gender"]:
            self.gender_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM // 2, 2)
            )
        
        if model_type in ["unified", "teacher_temporal"]:
            self.temporal_head = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM // 2, 10)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        modules = [self.shared_projection]
        
        if hasattr(self, 'gender_head'):
            modules.append(self.gender_head)
        if hasattr(self, 'temporal_head'):
            modules.append(self.temporal_head)
        if hasattr(self, 'feature_attention'):
            modules.append(self.feature_attention)
        
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, feature_annealing_weight=1.0):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        if config.FEATURE_ANNEALING and hasattr(self, 'feature_attention'):
            attention_weights = self.feature_attention(cls_output)
            cls_output = cls_output * (attention_weights * feature_annealing_weight +
                                     (1 - feature_annealing_weight))
        
        cls_output = self.dropout(cls_output)
        
        shared_repr = self.shared_projection(cls_output)
        
        result = {'shared_representation': shared_repr}
        
        if hasattr(self, 'gender_head'):
            result['gender_logits'] = self.gender_head(shared_repr)
        
        if hasattr(self, 'temporal_head'):
            result['temporal_logits'] = self.temporal_head(shared_repr)
        
        if config.FEATURE_ANNEALING and hasattr(self, 'feature_attention'):
            result['attention_weights'] = attention_weights
        
        return result

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, true_labels, teacher_weight=1.0):
        classification_loss = self.ce_loss(student_logits, true_labels)
        
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        total_loss = (
            (1 - self.alpha) * classification_loss +
            self.alpha * teacher_weight * distillation_loss
        )
        
        return total_loss, classification_loss, distillation_loss

class Trainer:
    def __init__(self, device):
        self.device = device
        self.training_history = defaultdict(list)
    
    def compute_annealing_weights(self, epoch, total_epochs):
        progress = epoch / total_epochs
        
        if config.TEACHER_ANNEALING:
            if progress < config.ANNEALING_START:
                teacher_weight = 1.0
            elif progress > config.ANNEALING_END:
                teacher_weight = config.ANNEALING_MIN_WEIGHT
            else:
                annealing_progress = (progress - config.ANNEALING_START) / (
                    config.ANNEALING_END - config.ANNEALING_START)
                teacher_weight = 1.0 - (1.0 - config.ANNEALING_MIN_WEIGHT) * annealing_progress
        else:
            teacher_weight = 1.0
        
        if config.FEATURE_ANNEALING:
            if progress < config.ATTENTION_ANNEALING_START:
                feature_weight = 1.0
            elif progress > config.ATTENTION_ANNEALING_END:
                feature_weight = 0.3
            else:
                annealing_progress = (progress - config.ATTENTION_ANNEALING_START) / (
                    config.ATTENTION_ANNEALING_END - config.ATTENTION_ANNEALING_START)
                feature_weight = 1.0 - 0.7 * annealing_progress
        else:
            feature_weight = 1.0
        
        return teacher_weight, feature_weight
    
    def train_teacher(self, model, train_loader, val_loader, task_type, num_epochs):
        print("\nEntraînement teacher {}".format(task_type.upper()))
        
        model.to(self.device)
        
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc="Teacher {} Epoch {}".format(task_type, epoch+1))
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if task_type == "gender":
                    labels = batch['gender_label'].to(self.device)
                    logits_key = "gender_logits"
                else:
                    labels = batch['temporal_label'].to(self.device)
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
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'Loss': '{:.4f}'.format(loss.item()), 
                                'Acc': '{:.1f}%'.format(100*correct/total)})
            
            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            if val_loader:
                val_acc = self.evaluate_model(model, val_loader, task_type)
                print("Epoch {}: Loss = {:.4f}, Train Acc = {:.1f}%, Val Acc = {:.1f}%".format(
                    epoch+1, avg_loss, train_acc, val_acc))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(config.MODELS_DIR, exist_ok=True)
                    torch.save(model.state_dict(), 
                              os.path.join(config.MODELS_DIR, "teacher_{}_best.pth".format(task_type)))
                    print("Teacher {} sauvé (Acc: {:.1f}%)".format(task_type, val_acc))
            else:
                print("Epoch {}: Loss = {:.4f}, Train Acc = {:.1f}%".format(epoch+1, avg_loss, train_acc))
        
        print("Teacher {} terminé! Meilleure accuracy: {:.1f}%".format(task_type, best_val_acc))
        return model
    
    def train_student(self, student_model, teacher_gender, teacher_temporal, 
                     train_loader, val_loader, num_epochs):
        print("\nEntraînement BAM student")
        print("Distillation: alpha={}, temperature={}".format(config.ALPHA_DISTILL, config.TEMPERATURE))
        print("Teacher Annealing: {:.1%} -> {:.1%}".format(config.ANNEALING_START, config.ANNEALING_END))
        print("Feature Annealing: {:.1%} -> {:.1%}".format(config.ATTENTION_ANNEALING_START, config.ATTENTION_ANNEALING_END))
        
        student_model.to(self.device)
        teacher_gender.to(self.device)
        teacher_temporal.to(self.device)
        
        teacher_gender.eval()
        teacher_temporal.eval()
        for param in teacher_gender.parameters():
            param.requires_grad = False
        for param in teacher_temporal.parameters():
            param.requires_grad = False
        
        optimizer = AdamW(student_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
            num_training_steps=total_steps
        )
        
        gender_distill_loss = DistillationLoss(config.TEMPERATURE, config.ALPHA_DISTILL)
        temporal_distill_loss = DistillationLoss(config.TEMPERATURE, config.ALPHA_DISTILL)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            teacher_weight, feature_weight = self.compute_annealing_weights(epoch, num_epochs)
            
            print("\nEpoch {}/{}".format(epoch+1, num_epochs))
            print("Teacher Weight: {:.3f}, Feature Weight: {:.3f}".format(teacher_weight, feature_weight))
            
            student_model.train()
            
            total_loss = 0
            gender_correct = 0
            temporal_correct = 0
            total_samples = 0
            
            pbar = tqdm(train_loader, desc="BAM Epoch {}".format(epoch+1))
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                temporal_labels = batch['temporal_label'].to(self.device)
                
                optimizer.zero_grad()
                
                student_outputs = student_model(input_ids, attention_mask,
                                               feature_annealing_weight=feature_weight)
                
                with torch.no_grad():
                    teacher_gender_outputs = teacher_gender(input_ids, attention_mask)
                    teacher_temporal_outputs = teacher_temporal(input_ids, attention_mask)
                
                gender_loss, gender_ce, gender_kl = gender_distill_loss(
                    student_outputs['gender_logits'],
                    teacher_gender_outputs['gender_logits'],
                    gender_labels,
                    teacher_weight
                )
                
                temporal_loss, temporal_ce, temporal_kl = temporal_distill_loss(
                    student_outputs['temporal_logits'],
                    teacher_temporal_outputs['temporal_logits'],
                    temporal_labels,
                    teacher_weight
                )
                
                total_loss_batch = gender_loss + temporal_loss
                total_loss_batch.backward()
                
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                
                total_loss += total_loss_batch.item()
                
                _, gender_pred = torch.max(student_outputs['gender_logits'], 1)
                _, temporal_pred = torch.max(student_outputs['temporal_logits'], 1)
                
                gender_correct += (gender_pred == gender_labels).sum().item()
                temporal_correct += (temporal_pred == temporal_labels).sum().item()
                total_samples += gender_labels.size(0)
                
                pbar.set_postfix({
                    'Loss': '{:.4f}'.format(total_loss_batch.item()),
                    'Gender': '{:.1f}%'.format(100*gender_correct/total_samples),
                    'Temporal': '{:.1f}%'.format(100*temporal_correct/total_samples),
                    'TW': '{:.2f}'.format(teacher_weight),
                    'FW': '{:.2f}'.format(feature_weight)
                })
            
            avg_loss = total_loss / len(train_loader)
            train_gender_acc = 100 * gender_correct / total_samples
            train_temporal_acc = 100 * temporal_correct / total_samples
            
            if val_loader:
                val_acc_gender = self.evaluate_model(student_model, val_loader, "gender")
                val_acc_temporal = self.evaluate_model(student_model, val_loader, "temporal")
                val_acc_avg = (val_acc_gender + val_acc_temporal) / 2
                
                self.training_history['teacher_weights'].append(teacher_weight)
                self.training_history['feature_weights'].append(feature_weight)
                self.training_history['train_gender_acc'].append(train_gender_acc)
                self.training_history['train_temporal_acc'].append(train_temporal_acc)
                self.training_history['val_gender_acc'].append(val_acc_gender)
                self.training_history['val_temporal_acc'].append(val_acc_temporal)
                
                print("Train - Gender: {:.1f}%, Temporal: {:.1f}%".format(train_gender_acc, train_temporal_acc))
                print("Val - Gender: {:.1f}%, Temporal: {:.1f}%".format(val_acc_gender, val_acc_temporal))
                
                if val_acc_avg > best_val_acc:
                    best_val_acc = val_acc_avg
                    torch.save(student_model.state_dict(), 
                              os.path.join(config.MODELS_DIR, "bam_student_best.pth"))
                    print("BAM Student sauvé (Avg: {:.1f}%)".format(val_acc_avg))
            else:
                print("Epoch {}: Loss = {:.4f}".format(epoch+1, avg_loss))
        
        print("BAM Student terminé! Meilleure accuracy: {:.1f}%".format(best_val_acc))
        return student_model
    
    def evaluate_model(self, model, dataloader, task_type):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if task_type == "gender":
                    labels = batch['gender_label'].to(self.device)
                    logits_key = "gender_logits"
                else:
                    labels = batch['temporal_label'].to(self.device)
                    logits_key = "temporal_logits"
                
                outputs = model(input_ids, attention_mask)
                logits = outputs[logits_key]
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        return accuracy

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_test_set(self, test_chunks):
        print("\nÉvaluation sur test set")
        
        test_dataset = TextDataset(test_chunks, self.tokenizer, config.MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        all_gender_preds = []
        all_temporal_preds = []
        all_gender_labels = []
        all_temporal_labels = []
        all_book_ids = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Évaluation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                _, gender_pred = torch.max(outputs['gender_logits'], 1)
                _, temporal_pred = torch.max(outputs['temporal_logits'], 1)
                
                all_gender_preds.extend(gender_pred.cpu().numpy())
                all_temporal_preds.extend(temporal_pred.cpu().numpy())
                all_gender_labels.extend(batch['gender_label'].numpy())
                all_temporal_labels.extend(batch['temporal_label'].numpy())
                all_book_ids.extend(batch['book_id'])
        
        chunk_gender_acc = accuracy_score(all_gender_labels, all_gender_preds)
        chunk_temporal_acc = accuracy_score(all_temporal_labels, all_temporal_preds)
        
        print("CHUNKS - Genre: {:.1%}, Temporel: {:.1%}".format(chunk_gender_acc, chunk_temporal_acc))
        
        gender_ci = self.bootstrap_ci(all_gender_labels, all_gender_preds)
        temporal_ci = self.bootstrap_ci(all_temporal_labels, all_temporal_preds)
        
        print("IC 95% - Genre: [{:.1%}, {:.1%}]".format(gender_ci[1], gender_ci[2]))
        print("IC 95% - Temporel: [{:.1%}, {:.1%}]".format(temporal_ci[1], temporal_ci[2]))
        
        book_results = self.evaluate_by_book(test_chunks, all_gender_preds, all_temporal_preds)
        
        book_gender_acc = 0
        book_temporal_acc = 0
        if book_results:
            book_gender_acc = np.mean([r['gender_correct'] for r in book_results])
            book_temporal_acc = np.mean([r['temporal_correct'] for r in book_results])
            
            print("LIVRES - Genre: {:.1%}, Temporel: {:.1%}".format(book_gender_acc, book_temporal_acc))
            print("Livres évalués: {}".format(len(book_results)))
        
        return {
            'chunk_gender_acc': chunk_gender_acc,
            'chunk_temporal_acc': chunk_temporal_acc,
            'gender_ci': gender_ci,
            'temporal_ci': temporal_ci,
            'book_results': book_results,
            'book_gender_acc': book_gender_acc,
            'book_temporal_acc': book_temporal_acc
        }
    
    def bootstrap_ci(self, y_true, y_pred, n_bootstrap=1000):
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_y_true = np.array(y_true)[indices]
            bootstrap_y_pred = np.array(y_pred)[indices]
            
            score = accuracy_score(bootstrap_y_true, bootstrap_y_pred)
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        mean_score = np.mean(bootstrap_scores)
        lower = np.percentile(bootstrap_scores, 2.5)
        upper = np.percentile(bootstrap_scores, 97.5)
        
        return mean_score, lower, upper
    
    def evaluate_by_book(self, test_chunks, gender_preds, temporal_preds):
        book_data = defaultdict(list)
        
        for i, chunk in enumerate(test_chunks):
            book_data[chunk['book_id']].append({
                'true_gender': chunk['sexe'],
                'true_temporal': chunk['classe_temporelle'],
                'pred_gender': gender_preds[i],
                'pred_temporal': temporal_preds[i]
            })
        
        book_results = []
        
        for book_id, book_chunks in book_data.items():
            if len(book_chunks) < config.MIN_CHUNKS_PER_BOOK:
                continue
            
            true_gender = book_chunks[0]['true_gender']
            true_temporal = book_chunks[0]['true_temporal']
            
            gender_votes = [chunk['pred_gender'] for chunk in book_chunks]
            temporal_votes = [chunk['pred_temporal'] for chunk in book_chunks]
            
            final_gender = max(set(gender_votes), key=gender_votes.count)
            final_temporal = max(set(temporal_votes), key=temporal_votes.count)
            
            book_results.append({
                'book_id': book_id,
                'num_chunks': len(book_chunks),
                'true_gender': true_gender,
                'true_temporal': true_temporal,
                'pred_gender': final_gender,
                'pred_temporal': final_temporal,
                'gender_correct': final_gender == true_gender,
                'temporal_correct': final_temporal == true_temporal
            })
        
        return book_results

def main():
    print("Pipeline principal")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    data = load_data()
    if not data:
        raise ValueError("Impossible de charger les données")
    
    train_chunks, val_chunks, test_chunks = split_by_books(data)
    
    print("\nInitialisation BERT + BAM")
    
    try:
        if 'camembert' in config.BERT_MODEL:
            tokenizer = CamembertTokenizer.from_pretrained(config.BERT_MODEL)
        elif 'flaubert' in config.BERT_MODEL:
            tokenizer = FlaubertTokenizer.from_pretrained(config.BERT_MODEL)
        else:
            tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        print("Tokenizer: {}".format(config.BERT_MODEL))
    except Exception as e:
        print("Erreur tokenizer: {}".format(e))
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    train_dataset = TextDataset(train_chunks, tokenizer, config.MAX_LENGTH)
    val_dataset = TextDataset(val_chunks, tokenizer, config.MAX_LENGTH) if val_chunks else None
    test_dataset = TextDataset(test_chunks, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=2) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=2)
    
    print("\nDataLoaders créés:")
    print("Train: {} batches".format(len(train_loader)))
    print("Val: {} batches".format(len(val_loader) if val_loader else 0))
    print("Test: {} batches".format(len(test_loader)))
    
    trainer = Trainer(device)
    
    print("\n" + "="*60)
    print("PHASE 1: Entraînement des teachers")
    print("="*60)
    
    teacher_gender = BertBAM(config.BERT_MODEL, "teacher_gender")
    teacher_gender = trainer.train_teacher(
        teacher_gender, train_loader, val_loader, "gender", config.NUM_EPOCHS_TEACHER
    )
    
    teacher_temporal = BertBAM(config.BERT_MODEL, "teacher_temporal")
    teacher_temporal = trainer.train_teacher(
        teacher_temporal, train_loader, val_loader, "temporal", config.NUM_EPOCHS_TEACHER
    )
    
    print("\n" + "="*60)
    print("PHASE 2: Entraînement du student")
    print("="*60)
    
    student_model = BertBAM(config.BERT_MODEL, "unified")
    student_model = trainer.train_student(
        student_model, teacher_gender, teacher_temporal,
        train_loader, val_loader, config.NUM_EPOCHS_STUDENT
    )
    
    print("\n" + "="*60)
    print("PHASE 3: Évaluation")
    print("="*60)
    
    try:
        student_model.load_state_dict(
            torch.load(os.path.join(config.MODELS_DIR, "bam_student_best.pth"))
        )
        print("Meilleur modèle student chargé")
    except:
        print("Modèle par défaut utilisé")
    
    evaluator = Evaluator(student_model, tokenizer, device)
    results = evaluator.evaluate_test_set(test_chunks)
    
    print("\n" + "="*80)
    print("RAPPORT FINAL")
    print("="*80)
    
    print("\nDonnées:")
    print("Train: {} chunks".format(len(train_chunks)))
    print("Val: {} chunks".format(len(val_chunks)))
    print("Test: {} chunks".format(len(test_chunks)))
    
    print("\nPerformance (chunks):")
    print("Genre: {:.1%} [IC: {:.1%}-{:.1%}]".format(
        results['chunk_gender_acc'], results['gender_ci'][1], results['gender_ci'][2]))
    print("Temporel: {:.1%} [IC: {:.1%}-{:.1%}]".format(
        results['chunk_temporal_acc'], results['temporal_ci'][1], results['temporal_ci'][2]))
    
    if results['book_results']:
        print("\nPerformance (livres):")
        print("Genre: {:.1%}".format(results['book_gender_acc']))
        print("Temporel: {:.1%}".format(results['book_temporal_acc']))
 
    temporal_acc_final = results['book_temporal_acc'] if results['book_results'] else results['chunk_temporal_acc']
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        'timestamp': timestamp,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')},
        'results': results,
        'training_history': trainer.training_history,
    }
    
    try:
        results_file = os.path.join(config.RESULTS_DIR, 'bam_results_{}.json'.format(timestamp))
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, default=str, ensure_ascii=False)
        print("\nRésultats sauvés: {}".format(results_file))
    except Exception as e:
        print("Erreur sauvegarde: {}".format(e))
    
    print("Performance temporelle finale: {:.1%}".format(temporal_acc_final))
    
    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nErreur: {}".format(e))
        import traceback
        traceback.print_exc()
