"""
Évaluation et visualisation du projet BAM
Ce script permet d'évaluer les modèles entraînés et de générer des visualisations
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

from bam_project import BertBAM, TextDataset, Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EvaluationConfig:
    DATA_DIR = '/content/preprocessed_data'
    MODELS_DIR = '/content/models'
    RESULTS_DIR = '/content/results'
    PLOTS_DIR = '/content/plots'
    
    BATCH_SIZE = 16
    TOLERANCE_YEARS = 25
    
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
        print("Données de test chargées: {} chunks".format(len(test_data)))
        return test_data
    else:
        print("Fichier de test non trouvé, recherche alternative...")
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
                else:
                    model = BertBAM('camembert-base', "teacher_temporal")
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models[model_name] = model
                print("Modèle {} chargé avec succès".format(model_name))
            except Exception as e:
                print("Erreur lors du chargement de {}: {}".format(model_name, e))
        else:
            print("Modèle {} non trouvé: {}".format(model_name, model_path))
    
    return models

def predict_chunks(model, dataloader, model_type):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Prédiction {}".format(model_type)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gender_labels = batch['gender_label'].to(device)
            temporal_labels = batch['temporal_label'].to(device)
            book_ids = batch['book_id']
            
            outputs = model(input_ids, attention_mask)
            
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
                
            else:
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
        true_gender = chunks[0]['gender_label']
        true_temporal = chunks[0]['temporal_label']
        
        gender_votes = [chunk['gender_pred'] for chunk in chunks]
        temporal_votes = [chunk['temporal_pred'] for chunk in chunks]
        
        pred_gender_vote = max(set(gender_votes), key=gender_votes.count)
        pred_temporal_vote = max(set(temporal_votes), key=temporal_votes.count)
        
        gender_probs_avg = np.mean([chunk['gender_probs'] for chunk in chunks], axis=0)
        temporal_probs_avg = np.mean([chunk['temporal_probs'] for chunk in chunks], axis=0)
        
        pred_gender_softmax = np.argmax(gender_probs_avg)
        pred_temporal_softmax = np.argmax(temporal_probs_avg)
        
        temporal_centers_sum = 0.0
        for chunk in chunks:
            pred_class = chunk['temporal_pred']
            pred_year = eval_config.TEMPORAL_MIDPOINTS[pred_class]
            temporal_centers_sum += pred_year
        
        temporal_mean_centers = temporal_centers_sum / len(chunks)
        pred_temporal_mean = year_to_class(temporal_mean_centers)
        
        temporal_weighted_sum = 0.0
        temporal_weights_sum = 0.0
        
        for chunk in chunks:
            pred_class = chunk['temporal_pred']
            pred_year = eval_config.TEMPORAL_MIDPOINTS[pred_class]
            confidence = chunk['temporal_probs'][pred_class]
            
            temporal_weighted_sum += pred_year * confidence
            temporal_weights_sum += confidence
        
        temporal_barycentre_year = temporal_weighted_sum / temporal_weights_sum if temporal_weights_sum > 0 else 1920
        pred_temporal_barycentre = year_to_class(temporal_barycentre_year)
        
        book_results.append({
            'book_id': book_id,
            'num_chunks': len(chunks),
            'true_gender': true_gender,
            'true_temporal': true_temporal,
            'pred_gender_vote': pred_gender_vote,
            'pred_temporal_vote': pred_temporal_vote,
            'pred_gender_softmax': pred_gender_softmax,
            'pred_temporal_softmax': pred_temporal_softmax,
            'pred_temporal_mean': pred_temporal_mean,
            'temporal_mean_centers': temporal_mean_centers,
            'pred_temporal_barycentre': pred_temporal_barycentre,
            'temporal_barycentre_year': temporal_barycentre_year,
            'gender_confidence': np.max(gender_probs_avg),
            'temporal_confidence': np.max(temporal_probs_avg),
            'avg_gender_confidence': np.mean([chunk['gender_probs'][chunk['gender_pred']] for chunk in chunks]),
            'avg_temporal_confidence': np.mean([chunk['temporal_probs'][chunk['temporal_pred']] for chunk in chunks])
        })
    
    return book_results

def calculate_temporal_accuracy_with_tolerance(book_results, tolerance_years=25, method='vote'):
    correct_strict = 0
    correct_tolerant = 0
    total = len(book_results)
    
    if method == 'vote':
        pred_key = 'pred_temporal_vote'
    elif method == 'mean':
        pred_key = 'pred_temporal_mean'
    elif method == 'barycentre':
        pred_key = 'pred_temporal_barycentre'
    else:
        pred_key = 'pred_temporal_vote'
    
    for result in book_results:
        true_year = eval_config.TEMPORAL_MIDPOINTS[result['true_temporal']]
        
        if method == 'mean':
            pred_year = result['temporal_mean_centers']
        elif method == 'barycentre':
            pred_year = result['temporal_barycentre_year']
        else:
            pred_year = eval_config.TEMPORAL_MIDPOINTS[result[pred_key]]
        
        if result['true_temporal'] == result[pred_key]:
            correct_strict += 1
        
        if abs(true_year - pred_year) <= tolerance_years:
            correct_tolerant += 1
    
    return correct_strict / total, correct_tolerant / total

def compare_aggregation_methods(book_results):
    print("\nComparaison des méthodes d'agrégation:")
    print("=" * 60)
    
    gender_methods = {
        'Vote majoritaire': 'vote'
    }
    
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
        
        print("  {}: {:.4f}".format(method_name, gender_accuracy))
    
    print("\nTEMPOREL (toutes les méthodes):")
    for method_name, method_key in temporal_methods.items():
        temporal_acc_strict, temporal_acc_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, 25, method_key
        )
        
        results_comparison[method_name] = {
            'temporal_accuracy_strict': temporal_acc_strict,
            'temporal_accuracy_tolerant': temporal_acc_tolerant
        }
        
        print("  {}:".format(method_name))
        print("    Strict: {:.4f}".format(temporal_acc_strict))
        print("    ±25 ans: {:.4f}".format(temporal_acc_tolerant))
    
    best_method_strict = max(results_comparison.items(), 
                           key=lambda x: x[1]['temporal_accuracy_strict'])
    best_method_tolerant = max(results_comparison.items(), 
                             key=lambda x: x[1]['temporal_accuracy_tolerant'])
    
    print("\nMeilleure méthode temporelle (strict): {} ({:.4f})".format(
        best_method_strict[0], best_method_strict[1]['temporal_accuracy_strict']))
    print("Meilleure méthode temporelle (±25 ans): {} ({:.4f})".format(
        best_method_tolerant[0], best_method_tolerant[1]['temporal_accuracy_tolerant']))
    
    return results_comparison

def create_visualizations(book_results, model_name):
    print("Création des visualisations pour {}...".format(model_name))
    
    os.makedirs(eval_config.PLOTS_DIR, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Résultats du modèle {}'.format(model_name), fontsize=16, fontweight='bold')
    
    true_gender = [r['true_gender'] for r in book_results]
    pred_gender = [r['pred_gender_vote'] for r in book_results]
    
    cm_gender = confusion_matrix(true_gender, pred_gender)
    sns.heatmap(cm_gender, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Femme', 'Homme'], yticklabels=['Femme', 'Homme'],
                ax=axes[0,0])
    axes[0,0].set_title('Matrice de confusion - Genre (Vote majoritaire)')
    axes[0,0].set_xlabel('Prédiction')
    axes[0,0].set_ylabel('Vérité')
    
    true_temporal = [r['true_temporal'] for r in book_results]
    pred_temporal = [r['pred_temporal_softmax'] for r in book_results]
    
    cm_temporal = confusion_matrix(true_temporal, pred_temporal)
    sns.heatmap(cm_temporal, annot=True, fmt='d', cmap='Reds', ax=axes[0,1])
    axes[0,1].set_title('Matrice de confusion - Période temporelle')
    axes[0,1].set_xlabel('Prédiction')
    axes[0,1].set_ylabel('Vérité')
    
    gender_confidences = [r['gender_confidence'] for r in book_results]
    axes[1,0].hist(gender_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].set_title('Distribution des confiances - Genre')
    axes[1,0].set_xlabel('Confiance')
    axes[1,0].set_ylabel('Nombre de livres')
    axes[1,0].axvline(np.mean(gender_confidences), color='red', linestyle='--', 
                      label='Moyenne: {:.3f}'.format(np.mean(gender_confidences)))
    axes[1,0].legend()
    
    temporal_confidences = [r['temporal_confidence'] for r in book_results]
    axes[1,1].hist(temporal_confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,1].set_title('Distribution des confiances - Temporel')
    axes[1,1].set_xlabel('Confiance')
    axes[1,1].set_ylabel('Nombre de livres')
    axes[1,1].axvline(np.mean(temporal_confidences), color='red', linestyle='--',
                      label='Moyenne: {:.3f}'.format(np.mean(temporal_confidences)))
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_config.PLOTS_DIR, '{}_evaluation.png'.format(model_name)), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
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
    plt.title('Accuracy par période temporelle - {}'.format(model_name))
    plt.xticks(range(len(periods)), period_labels, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, '{:.2f}'.format(acc), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_config.PLOTS_DIR, '{}_temporal_accuracy.png'.format(model_name)), 
                dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_models():
    print("Début de l'évaluation complète...")
    
    test_data = load_test_data()
    if not test_data:
        print("Aucune donnée de test trouvée!")
        return
    
    models = load_trained_models()
    if not models:
        print("Aucun modèle trouvé!")
        return
    
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    test_dataset = TextDataset(test_data, tokenizer, 256)
    test_loader = DataLoader(test_dataset, batch_size=eval_config.BATCH_SIZE)
    
    all_results = {}
    
    for model_name, model in models.items():
        print("\nÉvaluation du modèle {}...".format(model_name))
        
        predictions = predict_chunks(model, test_loader, model_name)
        book_results = aggregate_by_book(predictions)
        
        if model_name == "bam_student":
            aggregation_comparison = compare_aggregation_methods(book_results)
            all_results[model_name + '_aggregation_methods'] = aggregation_comparison
        
        gender_accuracy = accuracy_score(
            [r['true_gender'] for r in book_results],
            [r['pred_gender_vote'] for r in book_results]
        )
        
        temporal_accuracy_vote_strict, temporal_accuracy_vote_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'vote'
        )
        
        temporal_accuracy_mean_strict, temporal_accuracy_mean_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'mean'
        )
        
        temporal_accuracy_barycentre_strict, temporal_accuracy_barycentre_tolerant = calculate_temporal_accuracy_with_tolerance(
            book_results, eval_config.TOLERANCE_YEARS, 'barycentre'
        )
        
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
        
        print("Résultats {}:".format(model_name))
        print("  Genre (vote majoritaire): {:.4f}".format(gender_accuracy))
        print("  Temporel - Vote majoritaire: {:.4f} (±{} ans: {:.4f})".format(
            temporal_accuracy_vote_strict, eval_config.TOLERANCE_YEARS, temporal_accuracy_vote_tolerant))
        print("  Temporel - Moyenne simple: {:.4f} (±{} ans: {:.4f})".format(
            temporal_accuracy_mean_strict, eval_config.TOLERANCE_YEARS, temporal_accuracy_mean_tolerant))
        print("  Temporel - Barycentre pondéré: {:.4f} (±{} ans: {:.4f})".format(
            temporal_accuracy_barycentre_strict, eval_config.TOLERANCE_YEARS, temporal_accuracy_barycentre_tolerant))
        
        create_visualizations(book_results, model_name)
    
    os.makedirs(eval_config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(eval_config.RESULTS_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 60)
    
    for model_name, results in all_results.items():
        if not model_name.endswith('_aggregation_methods'):
            print("\n{}:".format(model_name))
            print("  Genre: {:.4f}".format(results['gender_accuracy']))
            print("  Temporel - Vote: {:.4f} (±{} ans: {:.4f})".format(
                results['temporal_accuracy_vote_strict'], eval_config.TOLERANCE_YEARS, 
                results['temporal_accuracy_vote_tolerant']))
            print("  Temporel - Moyenne: {:.4f} (±{} ans: {:.4f})".format(
                results['temporal_accuracy_mean_strict'], eval_config.TOLERANCE_YEARS,
                results['temporal_accuracy_mean_tolerant']))
            print("  Temporel - Barycentre: {:.4f} (±{} ans: {:.4f})".format(
                results['temporal_accuracy_barycentre_strict'], eval_config.TOLERANCE_YEARS,
                results['temporal_accuracy_barycentre_tolerant']))
    
    print("\nRésultats sauvegardés dans {}/".format(eval_config.RESULTS_DIR))
    print("Visualisations sauvegardées dans {}/".format(eval_config.PLOTS_DIR))

def main():
    print("Démarrage de l'évaluation BAM")
    
    test_data = load_test_data()
    if test_data:
        print("Données réelles trouvées, lancement de l'évaluation complète")
        evaluate_all_models()
    else:
        print("Aucune donnée de test trouvée")

if __name__ == "__main__":
    main()
