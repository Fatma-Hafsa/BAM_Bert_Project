"""
Préprocessing des données textuelles pour le projet de classification avec CamemBERT.
Gestion de la tokenisation, création des chunks, et splits stratifiés.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import CamembertTokenizer
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import pickle
from collections import defaultdict

class TextPreprocessor:
    
    def __init__(self, corpus_path, metadata_path, max_length=512, overlap=50):
        
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        self.max_length = max_length
        self.overlap = overlap
        
        # Initialiser le tokenizer CamemBERT
        print("Chargement du tokenizer CamemBERT...")
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        
        # Charger les métadonnées
        print("Chargement des métadonnées...")
        self.metadata = pd.read_csv(metadata_path)
        
        # Ajuster les classes temporelles pour avoir exactement 10 classes
        self.metadata = self._adjust_temporal_classes()
        
        print(f"Corpus chargé: {len(self.metadata)} textes")
        print(f"Classes temporelles: {sorted(self.metadata['classe_temporelle'].unique())}")
        
    def _adjust_temporal_classes(self):
        df = self.metadata.copy()
        
        # Filtrer pour garder seulement les livres avec date_publication <= 2020
        df = df[df['date_publication'] <= 2020].copy()
        
        # Recalculer les classes temporelles pour s'assurer qu'elles sont correctes
        def get_temporal_class(year):
            if year < 1820 or year > 2020:
                return None
            return int((year - 1820) // 20)
        
        df['classe_temporelle'] = df['date_publication'].apply(get_temporal_class)
        
        # Filtrer les classes valides (0-9 pour 1820-2020)
        df = df[df['classe_temporelle'].between(0, 9)].copy()
        
        print(f"Après filtrage temporel (≤2020): {len(df)} textes")
        print("Distribution des classes temporelles:")
        for classe in sorted(df['classe_temporelle'].unique()):
            count = (df['classe_temporelle'] == classe).sum()
            start_year = 1820 + classe * 20
            end_year = min(start_year + 19, 2020)  # La dernière classe va jusqu'à 2020
            print(f"  Classe {classe} ({start_year}-{end_year}): {count} textes")
            
        return df
    
    def load_text_content(self, filename):
        filepath = os.path.join(self.corpus_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")
            return None
    
    def clean_text(self, text):
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', ' ', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les espaces en début/fin
        text = text.strip()
        
        return text
    
    def create_chunks(self, text, book_id):
    
        if not text:
            return []
        
        # Nettoyer le texte
        text = self.clean_text(text)
        
        # Tokeniser le texte complet
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) == 0:
            return []
        
        chunks = []
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(tokens):
            # Définir la fin du chunk
            end_idx = min(start_idx + self.max_length - 2, len(tokens))  # -2 pour [CLS] et [SEP]
            
            # Extraire les tokens du chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Convertir en texte
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            
            # Créer l'entrée du chunk
            chunk_data = {
                'book_id': book_id,
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_token': start_idx,
                'end_token': end_idx,
                'num_tokens': len(chunk_tokens)
            }
            
            chunks.append(chunk_data)
            
            # Calculer le prochain point de départ avec overlap
            if end_idx >= len(tokens):
                break
                
            start_idx = end_idx - self.overlap
            chunk_id += 1
        
        return chunks
    
    def process_all_texts(self):
        print("Traitement de tous les textes...")
        
        all_chunks = []
        failed_files = []
        
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            filename = row['filename']
            
            # Charger le contenu du texte
            content = self.load_text_content(filename)
            
            if content is None:
                failed_files.append(filename)
                continue
            
            # Créer les chunks
            chunks = self.create_chunks(content, idx)
            
            # Ajouter les métadonnées à chaque chunk
            for chunk in chunks:
                chunk.update({
                    'sexe': row['sexe_num'] - 1,  # Convertir 1,2 en 0,1
                    'sexe_label': row['sexe_label'],
                    'classe_temporelle': int(row['classe_temporelle']),
                    'date_publication': row['date_publication'],
                    'auteur': f"{row['prenom']} {row['nom']}",
                    'titre': row['titre'],
                    'filename': filename
                })
                
            all_chunks.extend(chunks)
        
        if failed_files:
            print(f"Échec du chargement de {len(failed_files)} fichiers:")
            for f in failed_files[:5]:  # Afficher les 5 premiers
                print(f"  - {f}")
            if len(failed_files) > 5:
                print(f"  ... et {len(failed_files) - 5} autres")
        
        print(f"Total de chunks créés: {len(all_chunks)}")
        
        return all_chunks
    
    def create_stratified_splits(self, chunks, test_size=0.2, val_size=0.2, random_state=42):
        
        print("Création des splits stratifiés...")
        
        # Créer un DataFrame des chunks
        df_chunks = pd.DataFrame(chunks)
        
        # Grouper par livre pour faire le split au niveau livre
        book_metadata = df_chunks.groupby('book_id').agg({
            'sexe': 'first',
            'classe_temporelle': 'first',
            'filename': 'first',
            'auteur': 'first'
        }).reset_index()
        
        print(f"Nombre de livres uniques: {len(book_metadata)}")
        
        # Créer une clé de stratification combinant sexe et classe temporelle
        book_metadata['strat_key'] = book_metadata['sexe'].astype(str) + '_' + book_metadata['classe_temporelle'].astype(str)
        
        # Vérifier la distribution des clés de stratification
        strat_counts = book_metadata['strat_key'].value_counts()
        print("Distribution des combinaisons sexe×classe:")
        for key, count in strat_counts.items():
            sexe, classe = key.split('_')
            sexe_label = 'Femme' if int(sexe) == 1 else 'Homme'
            start_year = 1820 + int(classe) * 20
            end_year = start_year + 19
            print(f"  {sexe_label} × {start_year}-{end_year}: {count} livres")
        
        # Premier split: train+val vs test
        train_val_books, test_books = train_test_split(
            book_metadata,
            test_size=test_size,
            stratify=book_metadata['strat_key'],
            random_state=random_state
        )
        
        # Deuxième split: train vs val
        train_books, val_books = train_test_split(
            train_val_books,
            test_size=val_size,
            stratify=train_val_books['strat_key'],
            random_state=random_state
        )
        
        print(f"Split des livres:")
        print(f"  Train: {len(train_books)} livres")
        print(f"  Validation: {len(val_books)} livres")
        print(f"  Test: {len(test_books)} livres")
        
        # Créer les sets de chunks correspondants
        train_book_ids = set(train_books['book_id'])
        val_book_ids = set(val_books['book_id'])
        test_book_ids = set(test_books['book_id'])
        
        train_chunks = [chunk for chunk in chunks if chunk['book_id'] in train_book_ids]
        val_chunks = [chunk for chunk in chunks if chunk['book_id'] in val_book_ids]
        test_chunks = [chunk for chunk in chunks if chunk['book_id'] in test_book_ids]
        
        print(f"Split des chunks:")
        print(f"  Train: {len(train_chunks)} chunks")
        print(f"  Validation: {len(val_chunks)} chunks")
        print(f"  Test: {len(test_chunks)} chunks")
        
        # Vérifier la distribution des classes dans chaque split
        for split_name, split_chunks in [('Train', train_chunks), ('Val', val_chunks), ('Test', test_chunks)]:
            split_df = pd.DataFrame(split_chunks)
            print(f"\nDistribution {split_name}:")
            print(f"  Sexe: {split_df['sexe'].value_counts().to_dict()}")
            print(f"  Classes temporelles: {split_df['classe_temporelle'].value_counts().sort_index().to_dict()}")
        
        return {
            'train': train_chunks,
            'val': val_chunks,
            'test': test_chunks,
            'train_books': train_books,
            'val_books': val_books,
            'test_books': test_books
        }
    
    def save_processed_data(self, splits, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder les splits
        for split_name, chunks in [('train', splits['train']), ('val', splits['val']), ('test', splits['test'])]:
            filepath = os.path.join(output_dir, f'{split_name}_chunks.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"Sauvegardé: {filepath}")
        
        # Sauvegarder les métadonnées des livres
        for split_name, books in [('train', splits['train_books']), ('val', splits['val_books']), ('test', splits['test_books'])]:
            filepath = os.path.join(output_dir, f'{split_name}_books.csv')
            books.to_csv(filepath, index=False)
            print(f"Sauvegardé: {filepath}")
        
        # Sauvegarder les statistiques
        stats = self.compute_statistics(splits)
        stats_filepath = os.path.join(output_dir, 'preprocessing_stats.json')
        with open(stats_filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Sauvegardé: {stats_filepath}")
        
        # Sauvegarder la configuration du tokenizer
        tokenizer_config = {
            'model_name': 'camembert-base',
            'max_length': self.max_length,
            'overlap': self.overlap,
            'vocab_size': self.tokenizer.vocab_size,
            'pad_token_id': self.tokenizer.pad_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id
        }
        
        config_filepath = os.path.join(output_dir, 'tokenizer_config.json')
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        print(f"Sauvegardé: {config_filepath}")
    
    def compute_statistics(self, splits):
        stats = {}
        
        for split_name, chunks in [('train', splits['train']), ('val', splits['val']), ('test', splits['test'])]:
            split_df = pd.DataFrame(chunks)
            
            stats[split_name] = {
                'num_chunks': len(chunks),
                'num_books': split_df['book_id'].nunique(),
                'avg_tokens_per_chunk': float(split_df['num_tokens'].mean()),
                'min_tokens_per_chunk': int(split_df['num_tokens'].min()),
                'max_tokens_per_chunk': int(split_df['num_tokens'].max()),
                'sexe_distribution': {int(k): int(v) for k, v in split_df['sexe'].value_counts().to_dict().items()},
                'temporal_distribution': {int(k): int(v) for k, v in split_df['classe_temporelle'].value_counts().sort_index().to_dict().items()}
            }
        
        # Statistiques globales
        all_chunks = splits['train'] + splits['val'] + splits['test']
        all_df = pd.DataFrame(all_chunks)
        
        stats['global'] = {
            'total_chunks': len(all_chunks),
            'total_books': all_df['book_id'].nunique(),
            'avg_chunks_per_book': float(len(all_chunks) / all_df['book_id'].nunique()),
            'tokenizer_config': {
                'model': 'camembert-base',
                'max_length': self.max_length,
                'overlap': self.overlap
            }
        }
        
        return stats

def main():
    
    # Chemins
    corpus_path = "/content/CORPUS POUR MULTIHEAD - février 2025"
    metadata_path = "/content/metadata_clean.csv"
    output_dir = "/content/preprocessed_data"
    
    # Initialiser le préprocesseur
    preprocessor = TextPreprocessor(
        corpus_path=corpus_path,
        metadata_path=metadata_path,
        max_length=512,
        overlap=50
    )
    
    # Traiter tous les textes
    chunks = preprocessor.process_all_texts()
    
    if len(chunks) == 0:
        print("Aucun chunk créé. Arrêt du traitement.")
        return
    
    # Créer les splits stratifiés
    splits = preprocessor.create_stratified_splits(chunks)
    
    # Sauvegarder les données
    preprocessor.save_processed_data(splits, output_dir)
    
    print(f"\nPréprocessing terminé avec succès!")
    print(f"Données sauvegardées dans: {output_dir}")

if __name__ == "__main__":
    main()

