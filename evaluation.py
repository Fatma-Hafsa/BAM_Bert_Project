
"""
Évaluation et visualisation du projet BAM

Ce script permet d'évaluer les modèles entraînés et de générer des visualisations
pour analyser les performances de la classification multi-tâches.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CamembertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import du modèle principal
from bam_project import BertBAM, TextDataset, Config, device

class EvaluationConfig:
    
    # Chemins
    DATA_DIR = '/content/preprocessed_data'
    MODELS_DIR = '/content/models'
    RESULTS_DIR = '/content/results'
    PLOTS_DIR = '/content/plots'
    
    # Paramètres d'évaluation
    BATCH_SIZE = 16
    TOLERANCE_YEARS = 25  # Tolérance pour la datation
    
    # Périodes temporelles
    TEMPORAL_PERIODS = {
        0: "1820-1839", 1: "1840-1859", 2: "1860-1879", 3: "1880-1899", 4: "1900-1919",
        5: "1920-1939", 6: "1940-1959", 7: "1960-1979", 8: "1980-1999", 9: "2000-2020"
    }
    
    TEMPORAL_MIDPOINTS = {
        0: 1830, 1: 1850, 2: 1870, 3: 1890, 4: 1910,
        5: 1930, 6: 1950, 7: 1970, 8: 1990, 9: 2010
    }

eval_config = EvaluationConfig()

def load_test_data():
    print("Chargement des données de test...")
    
    test_file = os.path.join(eval_config.DATA_DIR, 'test_chunks.json')
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Données de test chargées: {len(test_data)} chunks")
        return test_data
    else:
        print("Fichier de test non trouvé, recherche alternative...")
        # Logique de recherche alternative ici
        return []

def load_trained_models():
    print("Chargement des modèles entraînés...")
    
    models = {}
    model_files = {
        "teacher_gender": "teacher_gender_best.pth",
        "teacher_temporal": "teacher_temporal_best.pth",
        "bam_student": "bam_student_best.pth"
    }
    
    for model_name, model_file in model_files.items():
        model_path = os.path.join(eval_config.MODELS_DIR, model_file)
        
        if os.path.exists(model_path):
            try:
                if model_name == "bam_student":
                    model = BertBAM('camembert-base', "unified")
                elif model_name == "teacher_gender":
                    model = BertBAM('camembert-base', "teacher_gender")
                else:  # teacher_temporal
                    model = BertBAM('camembert-base', "teacher_temporal")
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models[model_name] = model
                print(f"Modèle {model_name} chargé avec succès")
            except Exception as e:
                print(f"Erreur lors du chargement de {model_name}: {e}")
        else:
            print(f"Modèle {model_name} non trouvé: {model_path}")
    
    return models

def predict_chunks(model, dataloader, model_type):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Prédiction {model_type}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gender_labels = batch['gender_label'].to(device)
            temporal_labels = batch['temporal_label'].to(device)
            book_ids = batch['book_id']
            
            outputs = model(input_ids, attention_mask)
            
            # Prédictions selon le type de modèle
            if model_type == 'teacher_gender':
                if 'gender_logits' in outputs:
                    gender_preds = torch.argmax(outputs['gender_logits'], dim=1).cpu().numpy()
                    gender_probs = F.softmax(outputs['gender_logits'], dim=1).cpu().numpy()
                else:
                    gender_preds = np.zeros(len(book_ids), dtype=int)
                    gender_probs = np.zeros((len(book_ids), 2))
                
                temporal_preds = np.zeros(len(book_ids), dtype=int)
                temporal_probs = np.zeros((len(book_ids), 10))
                
            elif model_type == 'teacher_temporal':
                if 'temporal_logits' in outputs:
                    temporal_preds = torch.argmax(outputs['temporal_logits'], dim=1).cpu().numpy()
                    temporal_probs = F.softmax(outputs['temporal_logits'], dim=1).cpu().numpy()
                else:
                    temporal_preds = np.zeros(len(book_ids), dtype=int)
                    temporal_probs = np.zeros((len(book_ids), 10))
                
                gender_preds = np.zeros(len(book_ids), dtype=int)
                gender_probs = np.zeros((len(book_ids), 2))
                
            else:  # bam_student (unified)
                if 'gender_logits' in outputs:
                    gender_preds = torch.argmax(outputs['gender_logits'], dim=1).cpu().numpy()
                    gender_probs = F.softmax(outputs['gender_logits'], dim=1).cpu().numpy()
                else:
                    gender_preds = np.zeros(len(book_ids), dtype=int)
                    gender_probs = np.zeros((len(book_ids), 2))
                
                if 'temporal_logits' in outputs:
                    temporal_preds = torch.argmax(outputs['temporal_logits'], dim=1).cpu().numpy()
                    temporal_probs = F.softmax(outputs['temporal_logits'], dim=1).cpu().numpy()
                else:
                    temporal_preds = np.zeros(len(book_ids), dtype=int)
                    temporal_probs = np.zeros((len(book_ids), 10))
            
            # Stocker les résultats
            for i in range(len(book_ids)):
                predictions.append({
                    'book_id': book_ids[i],
                    'gender_label': gender_labels[i].item(),
                    'temporal_label': temporal_labels[i].item(),
                    'gender_pred': gender_preds[i],
                    'temporal_pred': temporal_preds[i],
                    'gender_probs': gender_probs[i],
                    'temporal_probs': temporal_probs[i]
                })
    
    return predictions

def year_to_class(year):
    if year < 1820:
        return 0
    elif year > 2020:
        return 9
    else:
        return min(9, max(0, int((year - 1820) // 20)))

def aggregate_by_book(predictions):
    print("Agrégation des prédictions par livre...")
    
    books_data = defaultdict(list)
    for pred in predictions:
        books_data[pred['book_id']].append(pred)
    
    book_results = []
    
    for book_id, chunks in books_data.items():
        # Labels vrais (identiques pour tous les chunks d'un livre)
        true_gender = chunks[0]['gender_label']
        true_temporal = chunks[0]['temporal_label']
        
        # 1. Agrégation par vote majoritaire
        gender_votes = [chunk['gender_pred'] for chunk in chunks]
        temporal_votes = [chunk['temporal_pred'] for chunk in chunks]
        
        pred_gender_vote = max(set(gender_votes), key=gender_votes.count)
        pred_temporal_vote = max(set(temporal_votes), key=temporal_votes.count)
        
        # 2. Agrégation par moyenne des probabilités (vote majoritaire pour softmax)
        gender_probs_avg = np.mean([chunk['gender_probs'] for chunk in chunks], axis=0)
        temporal_probs_avg = np.mean([chunk['temporal_probs'] for chunk in chunks], axis=0)
        
        pred_gender_softmax = np.argmax(gender_probs_avg)
        pred_temporal_softmax = np.argmax(temporal_probs_avg)
        
        # 3. Moyenne simple des centres prédits
        temporal_centers_sum = 0.0
        for chunk in chunks:
            pred_class = chunk['temporal_pred']
            pred_year = eval_config.TEMPORAL_MIDPOINTS[pred_class]
            temporal_centers_sum += pred_year
        
        temporal_mean_centers = temporal_centers_sum / len(chunks)
        pred_temporal_mean = year_to_class(temporal_mean_centers)
        
        # 4. Barycentre pondéré par confiance (SEULEMENT classe prédite)
        temporal_weighted_sum = 0.0
        temporal_weights_sum = 0.0
        
        for chunk in chunks:
            # Convertir la classe prédite en année (centre de la période)
            pred_class = chunk['temporal_pred']
            pred_year = eval_config.TEMPORAL_MIDPOINTS[pred_class]
            confidence = chunk['temporal_probs'][pred_class]  # Confiance de la classe prédite
            
            temporal_weighted_sum += pred_year * confidence
            temporal_weights_sum += confidence
        
        temporal_barycentre_year = temporal_weighted_sum / temporal_weights_sum if temporal_weights_sum > 0 else 1920
        pred_temporal_barycentre = year_to_class(temporal_barycentre_year)
        
        book_results.append({
            'book_id': book_id,
            'num_chunks': len(chunks),
            'true_gender': true_gender,
            'true_temporal': true_temporal,
            
            # Méthode 1: Vote majoritaire
            'pred_gender_vote': pred_gender_vote,
            'pred_temporal_vote': pred_temporal_vote,
            
            # Méthode 2: Moyenne softmax (pour compatibilité visualisations)
            'pred_gender_softmax': pred_gender_softmax,
            'pred_temporal_softmax': pred_temporal_softmax,
            
            # Méthode 3: Moyenne simple des centres (temporel seulement)
            'pred_temporal_mean': pred_temporal_mean,
            'temporal_mean_centers': temporal_mean_centers,
            
            # Méthode 4: Barycentre pondéré par confiance (temporel seulement)
            'pred_temporal_barycentre': pred_temporal_barycentre,
            'temporal_barycentre_year': temporal_barycentre_year,
            
            # Métriques de confiance
            'gender_confidence': np.max(gender_probs_avg),
            'temporal_confidence': np.max(temporal_probs_avg),
            'avg_gender_confidence': np.mean([chunk['gender_probs'][chunk['gender_pred']] for chunk in chunks]),
            'avg_temporal_confidence': np.mean([chunk['temporal_probs'][chunk['temporal_pred']] for chunk in chunks])
        })
    
    return book_results

def calculate_temporal_accuracy_with_tolerance(book_results, tolerance_years=25, method='softmax'):
    correct_strict = 0
    correct_tolerant = 0
    total = len(book_results)
    
    # Choisir la méthode de prédiction
    if method == 'vote':
        pred_key = 'pred_temporal_vote'
    elif method == 'mean':
        pred_key = 'pred_temporal_mean'
    elif method == 'barycentre':
        pred_key = 'pred_temporal_barycentre'
    else:
        pred_key = 'pred_temporal_vote'  # Par défaut
    
    for result in book_results:
        true_year = eval_config.TEMPORAL_MIDPOINTS[result['true_temporal']]
        
        if method == 'mean':
            # Pour la moyenne simple, on utilise directement l'année calculée
            pred_year = result['temporal_mean_centers']
        elif method == 'barycentre':
            # Pour le barycentre, on utilise directement l'année prédite
            pred_year = result['temporal_barycentre_year']
        else:
            # Pour vote, on utilise le centre de la classe prédite
            pred_year = eval_config.TEMPORAL_MIDPOINTS[result[pred_key]]
        
        # Accuracy stricte (classe exacte)
        if result['true_temporal'] == result[pred_key]:
            correct_strict += 1
        
        # Accuracy avec tolérance (années)
        if abs(true_year - pred_year) <= tolerance_years:
            correct_tolerant += 1
    
    return correct_strict / total, correct_tolerant / total

def compare_aggregation_methods(book_results):
    print("\nComparaison des méthodes d'agrégation:")
    print("="*60)
    
    # Pour le genre : seulement vote majoritaire
    gender_methods = {
        'Vote majoritaire': 'vote'
    }
    
    # Pour le temporel : seulement 3 méthodes
    temporal_methods = {
        'Vote majoritaire': 'vote',
        'Moyenne simple centres': 'mean',
        'Barycentre pondéré confiance': 'barycentre'
    }
    
    results_comparison = {}
    
    print("GENRE (vote majoritaire uniquement):")
    for method_name, method_key in gender_methods.items():
        gender_pred_key = 'pred_gender_vote'
        
        gender_accuracy = accuracy_score(
            [r['true_gender'] for r in book_results],
            [r[gender_pred_key] for r in book_results]
        )
        
        print(f"  {method_name}: {gender_accuracy:.4f}")
    
    print("\nTEMPOREL (toutes les méthodes):")
    for method_name, method_key in temporal_methods.items():
        # Accuracy temporelle
        temporal_acc_strict, temporal_acc_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, 25, method_key
        )
        
        results_comparison[method_name] = {
            'temporal_accuracy_strict': temporal_acc_strict,
            'temporal_accuracy_tolerant': temporal_acc_tolerant
        }
        
        print(f"  {method_name}:")
        print(f"    Strict: {temporal_acc_strict:.4f}")
        print(f"    ±25 ans: {temporal_acc_tolerant:.4f}")
    
    # Trouver la meilleure méthode pour le temporel
    best_method_strict = max(results_comparison.items(), 
                           key=lambda x: x[1]['temporal_accuracy_strict'])
    best_method_tolerant = max(results_comparison.items(), 
                             key=lambda x: x[1]['temporal_accuracy_tolerant'])
    
    print(f"\nMeilleure méthode temporelle (strict): {best_method_strict[0]} ({best_method_strict[1]['temporal_accuracy_strict']:.4f})")
    print(f"Meilleure méthode temporelle (±25 ans): {best_method_tolerant[0]} ({best_method_tolerant[1]['temporal_accuracy_tolerant']:.4f})")
    
    return results_comparison

def create_visualizations(book_results, model_name):
    print(f"Création des visualisations pour {model_name}...")
    
    os.makedirs(eval_config.PLOTS_DIR, exist_ok=True)
    
    # Configuration des plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Résultats du modèle {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Matrice de confusion - Genre (vote majoritaire)
    true_gender = [r['true_gender'] for r in book_results]
    pred_gender = [r['pred_gender_vote'] for r in book_results]
    
    cm_gender = confusion_matrix(true_gender, pred_gender)
    sns.heatmap(cm_gender, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Femme', 'Homme'], yticklabels=['Femme', 'Homme'],
                ax=axes[0,0])
    axes[0,0].set_title('Matrice de confusion - Genre (Vote majoritaire)')
    axes[0,0].set_xlabel('Prédiction')
    axes[0,0].set_ylabel('Vérité')
    
    # 2. Matrice de confusion - Temporel (méthode softmax)
    true_temporal = [r['true_temporal'] for r in book_results]
    pred_temporal = [r['pred_temporal_softmax'] for r in book_results]
    
    cm_temporal = confusion_matrix(true_temporal, pred_temporal)
    sns.heatmap(cm_temporal, annot=True, fmt='d', cmap='Reds', ax=axes[0,1])
    axes[0,1].set_title('Matrice de confusion - Période temporelle (Softmax)')
    axes[0,1].set_xlabel('Prédiction')
    axes[0,1].set_ylabel('Vérité')
    
    # 3. Distribution des confiances - Genre
    gender_confidences = [r['gender_confidence'] for r in book_results]
    axes[1,0].hist(gender_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].set_title('Distribution des confiances - Genre')
    axes[1,0].set_xlabel('Confiance')
    axes[1,0].set_ylabel('Nombre de livres')
    axes[1,0].axvline(np.mean(gender_confidences), color='red', linestyle='--', 
                      label=f'Moyenne: {np.mean(gender_confidences):.3f}')
    axes[1,0].legend()
    
    # 4. Distribution des confiances - Temporel
    temporal_confidences = [r['temporal_confidence'] for r in book_results]
    axes[1,1].hist(temporal_confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,1].set_title('Distribution des confiances - Temporel')
    axes[1,1].set_xlabel('Confiance')
    axes[1,1].set_ylabel('Nombre de livres')
    axes[1,1].axvline(np.mean(temporal_confidences), color='red', linestyle='--',
                      label=f'Moyenne: {np.mean(temporal_confidences):.3f}')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_config.PLOTS_DIR, f'{model_name}_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graphique séparé pour l'analyse temporelle détaillée
    plt.figure(figsize=(12, 8))
    
    # Accuracy par période (méthode softmax)
    period_accuracies = {}
    for period in range(10):
        period_books = [r for r in book_results if r['true_temporal'] == period]
        if period_books:
            correct = sum(1 for r in period_books if r['true_temporal'] == r['pred_temporal_softmax'])
            period_accuracies[period] = correct / len(period_books)
        else:
            period_accuracies[period] = 0
    
    periods = list(period_accuracies.keys())
    accuracies = list(period_accuracies.values())
    period_labels = [eval_config.TEMPORAL_PERIODS[p] for p in periods]
    
    plt.bar(range(len(periods)), accuracies, color='steelblue', alpha=0.7)
    plt.xlabel('Période temporelle')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy par période temporelle - {model_name}')
    plt.xticks(range(len(periods)), period_labels, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_config.PLOTS_DIR, f'{model_name}_temporal_accuracy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_models():
    print("Début de l'évaluation complète...")
    
    # Charger les données et modèles
    test_data = load_test_data()
    if not test_data:
        print("Aucune donnée de test trouvée!")
        return
    
    models = load_trained_models()
    if not models:
        print("Aucun modèle trouvé!")
        return
    
    # Préparer le dataset de test
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    test_dataset = TextDataset(test_data, tokenizer, 256)
    test_loader = DataLoader(test_dataset, batch_size=eval_config.BATCH_SIZE)
    
    # Évaluer chaque modèle
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nÉvaluation du modèle {model_name}...")
        
        # Prédictions
        predictions = predict_chunks(model, test_loader, model_name)
        
        # Agrégation par livre
        book_results = aggregate_by_book(predictions)
        
        # Comparaison des méthodes d'agrégation (pour le modèle principal seulement)
        if model_name == "bam_student":
            aggregation_comparison = compare_aggregation_methods(book_results)
            all_results[model_name + '_aggregation_methods'] = aggregation_comparison
        
        # Calcul des métriques avec différentes méthodes
        gender_accuracy = accuracy_score(
            [r['true_gender'] for r in book_results],
            [r['pred_gender_vote'] for r in book_results]  # Vote majoritaire pour le genre
        )
        
        # Métriques temporelles pour les 3 méthodes
        temporal_accuracy_vote_strict, temporal_accuracy_vote_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'vote'
        )
        
        temporal_accuracy_mean_strict, temporal_accuracy_mean_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'mean'
        )
        
        temporal_accuracy_barycentre_strict, temporal_accuracy_barycentre_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'barycentre'
        )
        
        # Stockage des résultats
        all_results[model_name] = {
            'gender_accuracy': gender_accuracy,
            'temporal_accuracy_vote_strict': temporal_accuracy_vote_strict,
            'temporal_accuracy_vote_tolerant': temporal_accuracy_vote_tolerant,
            'temporal_accuracy_mean_strict': temporal_accuracy_mean_strict,
            'temporal_accuracy_mean_tolerant': temporal_accuracy_mean_tolerant,
            'temporal_accuracy_barycentre_strict': temporal_accuracy_barycentre_strict,
            'temporal_accuracy_barycentre_tolerant': temporal_accuracy_barycentre_tolerant,
            'num_books': len(book_results),
            'avg_gender_confidence': np.mean([r['gender_confidence'] for r in book_results]),
            'avg_temporal_confidence': np.mean([r['temporal_confidence'] for r in book_results]),
            'avg_temporal_mean_year': np.mean([r['temporal_mean_centers'] for r in book_results]),
            'avg_temporal_barycentre_year': np.mean([r['temporal_barycentre_year'] for r in book_results])
        }
        
        print(f"Résultats {model_name}:")
        print(f"  Genre (vote majoritaire): {gender_accuracy:.4f}")
        print(f"  Temporel - Vote majoritaire: {temporal_accuracy_vote_strict:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {temporal_accuracy_vote_tolerant:.4f})")
        print(f"  Temporel - Moyenne simple: {temporal_accuracy_mean_strict:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {temporal_accuracy_mean_tolerant:.4f})")
        print(f"  Temporel - Barycentre pondéré: {temporal_accuracy_barycentre_strict:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {temporal_accuracy_barycentre_tolerant:.4f})")
        
        # Créer les visualisations
        create_visualizations(book_results, model_name)
    
    # Sauvegarder les résultats
    os.makedirs(eval_config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(eval_config.RESULTS_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Résumé comparatif
    print("\n" + "="*60)
    print("RÉSUMÉ COMPARATIF")
    print("="*60)
    
    for model_name, results in all_results.items():
        if not model_name.endswith('_aggregation_methods'):
            print(f"\n{model_name}:")
            print(f"  Genre: {results['gender_accuracy']:.4f}")
            print(f"  Temporel - Vote: {results['temporal_accuracy_vote_strict']:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {results['temporal_accuracy_vote_tolerant']:.4f})")
            print(f"  Temporel - Moyenne: {results['temporal_accuracy_mean_strict']:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {results['temporal_accuracy_mean_tolerant']:.4f})")
            print(f"  Temporel - Barycentre: {results['temporal_accuracy_barycentre_strict']:.4f} (±{eval_config.TOLERANCE_YEARS} ans: {results['temporal_accuracy_barycentre_tolerant']:.4f})")
    
    print(f"\nRésultats sauvegardés dans {eval_config.RESULTS_DIR}/")
    print(f"Visualisations sauvegardées dans {eval_config.PLOTS_DIR}/")

def main():
    print("DÉMARRAGE DE L'ÉVALUATION BAM")
    
    # Charger les données
    test_data = load_test_data()
    if test_data:
        print("Données réelles trouvées, lancement de l'évaluation complète")
        evaluate_all_models()
    else:
        print("Aucune donnée de test trouvée")

if __name__ == "__main__":
    main()

