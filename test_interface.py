"""
Interface de test sophistiquée pour le projet BAM
Permet de tester le modèle sur les livres du corpus avec navigation et comparaison
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import CamembertTokenizer
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bam_project import BertBAM, TextDataset, Config

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_data():
    print("Chargement des données de test...")
    
    test_file = os.path.join(config.DATA_DIR, 'test_chunks.json')
    if os.path.exists(test_file):
        import json
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print("Données de test chargées: {} chunks".format(len(test_data)))
        return test_data
    else:
        print("Fichier de test non trouvé")
        return []

def load_model():
    print("Chargement du modèle BAM...")
    
    model_path = os.path.join(config.MODELS_DIR, 'bam_student_best.pth')
    
    if not os.path.exists(model_path):
        print("Modèle non trouvé: {}".format(model_path))
        return None, None
    
    try:
        model = BertBAM(config.BERT_MODEL, "unified")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        tokenizer = CamembertTokenizer.from_pretrained(config.BERT_MODEL)
        
        print("Modèle chargé avec succès!")
        return model, tokenizer
        
    except Exception as e:
        print("Erreur lors du chargement: {}".format(e))
        return None, None

def predict_chunks(model, dataloader):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gender_labels = batch['gender_label'].to(device)
            temporal_labels = batch['temporal_label'].to(device)
            book_ids = batch['book_id']
            
            outputs = model(input_ids, attention_mask)
            
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
                all_predictions.append({
                    'book_id': book_ids[i],
                    'gender_label': gender_labels[i].item(),
                    'temporal_label': temporal_labels[i].item(),
                    'gender_pred': gender_preds[i],
                    'temporal_pred': temporal_preds[i],
                    'gender_probs': gender_probs[i].tolist(),
                    'temporal_probs': temporal_probs[i].tolist()
                })
    
    return all_predictions

def aggregate_by_book(predictions):
    books = defaultdict(list)
    for pred in predictions:
        books[pred['book_id']].append(pred)
    
    book_results = []
    
    for book_id, chunks in books.items():
        true_gender = chunks[0]['gender_label']
        true_temporal = chunks[0]['temporal_label']
        
        gender_preds = np.array([chunk['gender_pred'] for chunk in chunks])
        temporal_preds = np.array([chunk['temporal_pred'] for chunk in chunks])
        gender_probs = np.array([chunk['gender_probs'] for chunk in chunks])
        temporal_probs = np.array([chunk['temporal_probs'] for chunk in chunks])
        
        result = {
            'book_id': book_id,
            'true_gender': true_gender,
            'true_temporal': true_temporal,
            'num_chunks': len(chunks),
            'true_year': config.TEMPORAL_MIDPOINTS[true_temporal]
        }
        
        # 1. Vote majoritaire
        gender_votes = Counter(gender_preds)
        temporal_votes = Counter(temporal_preds)
        
        result['vote_gender'] = gender_votes.most_common(1)[0][0]
        result['vote_temporal'] = temporal_votes.most_common(1)[0][0]
        result['vote_year'] = config.TEMPORAL_MIDPOINTS[result['vote_temporal']]
        
        # 2. Moyenne des probabilités (barycentre softmax)
        gender_probs_mean = np.mean(gender_probs, axis=0)
        temporal_probs_mean = np.mean(temporal_probs, axis=0)
        
        result['barycentre_gender'] = np.argmax(gender_probs_mean)
        result['barycentre_temporal'] = np.argmax(temporal_probs_mean)
        
        temporal_weights = temporal_probs_mean
        estimated_year_barycentre = sum(w * config.TEMPORAL_MIDPOINTS[i] for i, w in enumerate(temporal_weights))
        result['barycentre_year'] = estimated_year_barycentre
        
        # 3. Moyenne des centres de classes
        estimated_year_centres = np.mean([config.TEMPORAL_MIDPOINTS[pred] for pred in temporal_preds])
        result['centres_year'] = estimated_year_centres
        result['centres_temporal'] = min(range(10), key=lambda i: abs(config.TEMPORAL_MIDPOINTS[i] - estimated_year_centres))
        result['centres_gender'] = int(round(np.mean(gender_preds)))
        
        # 4. Barycentre pondéré par confiance
        temporal_weighted_sum = 0.0
        temporal_weights_sum = 0.0
        
        for chunk in chunks:
            pred_class = chunk['temporal_pred']
            pred_year = config.TEMPORAL_MIDPOINTS[pred_class]
            confidence = chunk['temporal_probs'][pred_class]
            
            temporal_weighted_sum += pred_year * confidence
            temporal_weights_sum += confidence
        
        temporal_pondere_year = temporal_weighted_sum / temporal_weights_sum if temporal_weights_sum > 0 else 1920
        result['pondere_year'] = temporal_pondere_year
        result['pondere_temporal'] = min(range(10), key=lambda i: abs(config.TEMPORAL_MIDPOINTS[i] - temporal_pondere_year))
        
        book_results.append(result)
    
    return book_results

class BookTester:
    def __init__(self, model, tokenizer, test_data):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data
        
        # Organiser les données par livre
        self.books = defaultdict(list)
        for chunk in test_data:
            self.books[chunk['book_id']].append(chunk)
        
        self.book_ids = list(self.books.keys())
        print("Livres disponibles pour test: {}".format(len(self.book_ids)))
    
    def get_book_metadata(self, book_id):
        if book_id not in self.books:
            return None
        
        chunks = self.books[book_id]
        first_chunk = chunks[0]
        
        # Extraire le titre et l'auteur du nom de fichier si disponible
        title = "Titre inconnu"
        author = "Auteur inconnu"
        
        if 'filename' in first_chunk:
            parts = first_chunk['filename'].split(')')
            if len(parts) >= 2:
                author = parts[0].strip('()')
                title = parts[1].strip('()')
        elif 'titre' in first_chunk and 'auteur' in first_chunk:
            title = first_chunk['titre']
            author = first_chunk['auteur']
        
        return {
            'book_id': book_id,
            'title': title,
            'author': author,
            'num_chunks': len(chunks),
            'true_gender': first_chunk['sexe'],
            'true_temporal': first_chunk['classe_temporelle'],
            'true_gender_str': 'Femme' if first_chunk['sexe'] == 1 else 'Homme',
            'true_temporal_str': config.TEMPORAL_PERIODS[first_chunk['classe_temporelle']],
            'true_year': config.TEMPORAL_MIDPOINTS[first_chunk['classe_temporelle']]
        }
    
    def predict_book(self, book_id):
        if book_id not in self.books:
            print("Livre {} non trouvé".format(book_id))
            return None
        
        chunks = self.books[book_id]
        
        chunk_dataset = TextDataset(chunks, self.tokenizer, config.MAX_LENGTH)
        chunk_loader = DataLoader(chunk_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        predictions = predict_chunks(self.model, chunk_loader)
        book_results = aggregate_by_book(predictions)
        
        return book_results[0]
    
    def test_random_book(self):
        book_id = random.choice(self.book_ids)
        return self.test_book(book_id)
    
    def test_book(self, book_id):
        metadata = self.get_book_metadata(book_id)
        if metadata is None:
            print("Livre {} non trouvé".format(book_id))
            return None
        
        print("\nTest du livre")
        print("=" * 60)
        print("ID: {}".format(metadata['book_id']))
        print("Titre: {}".format(metadata['title']))
        print("Auteur: {}".format(metadata['author']))
        print("Chunks: {}".format(metadata['num_chunks']))
        print("Vérité - Genre: {}".format(metadata['true_gender_str']))
        print("Vérité - Époque: {} (année centrale: {})".format(metadata['true_temporal_str'], metadata['true_year']))
        
        print("\nPrédiction en cours...")
        result = self.predict_book(book_id)
        
        if result is None:
            print("Échec de la prédiction")
            return None
        
        print("\nRésultats:")
        print("-" * 60)
        
        # Méthode 1: Vote majoritaire
        if 'vote_gender' in result:
            gender_pred = 'Femme' if result['vote_gender'] == 1 else 'Homme'
            temporal_pred = config.TEMPORAL_PERIODS[result['vote_temporal']]
            gender_correct = result['vote_gender'] == metadata['true_gender']
            temporal_correct = result['vote_temporal'] == metadata['true_temporal']
            
            gender_status = "CORRECT" if gender_correct else "INCORRECT"
            temporal_status = "CORRECT" if temporal_correct else "INCORRECT"
            
            print("Vote majoritaire:")
            print("  Genre: {} ({})".format(gender_pred, gender_status))
            print("  Époque: {} ({})".format(temporal_pred, temporal_status))
            
            year_diff = abs(result['vote_year'] - metadata['true_year'])
            year_accurate = year_diff <= 25
            year_status = "CORRECT" if year_accurate else "INCORRECT"
            
            print("  Année estimée: {:.1f} (diff: {:.1f} ans) ({})".format(
                result['vote_year'], year_diff, year_status))
        
        # Méthode 2: Barycentre softmax
        if 'barycentre_gender' in result:
            gender_pred = 'Femme' if result['barycentre_gender'] == 1 else 'Homme'
            temporal_pred = config.TEMPORAL_PERIODS[result['barycentre_temporal']]
            gender_correct = result['barycentre_gender'] == metadata['true_gender']
            temporal_correct = result['barycentre_temporal'] == metadata['true_temporal']
            
            gender_status = "CORRECT" if gender_correct else "INCORRECT"
            temporal_status = "CORRECT" if temporal_correct else "INCORRECT"
            
            print("\nBarycentre softmax:")
            print("  Genre: {} ({})".format(gender_pred, gender_status))
            print("  Époque: {} ({})".format(temporal_pred, temporal_status))
            
            year_diff = abs(result['barycentre_year'] - metadata['true_year'])
            year_accurate = year_diff <= 25
            year_status = "CORRECT" if year_accurate else "INCORRECT"
            
            print("  Année estimée: {:.1f} (diff: {:.1f} ans) ({})".format(
                result['barycentre_year'], year_diff, year_status))
        
        # Méthode 3: Centres des classes
        if 'centres_gender' in result:
            gender_pred = 'Femme' if result['centres_gender'] == 1 else 'Homme'
            temporal_pred = config.TEMPORAL_PERIODS[result['centres_temporal']]
            gender_correct = result['centres_gender'] == metadata['true_gender']
            temporal_correct = result['centres_temporal'] == metadata['true_temporal']
            
            gender_status = "CORRECT" if gender_correct else "INCORRECT"
            temporal_status = "CORRECT" if temporal_correct else "INCORRECT"
            
            print("\nCentres des classes:")
            print("  Genre: {} ({})".format(gender_pred, gender_status))
            print("  Époque: {} ({})".format(temporal_pred, temporal_status))
            
            year_diff = abs(result['centres_year'] - metadata['true_year'])
            year_accurate = year_diff <= 25
            year_status = "CORRECT" if year_accurate else "INCORRECT"
            
            print("  Année estimée: {:.1f} (diff: {:.1f} ans) ({})".format(
                result['centres_year'], year_diff, year_status))
        
        # Méthode 4: Barycentre pondéré
        if 'pondere_temporal' in result:
            temporal_pred = config.TEMPORAL_PERIODS[result['pondere_temporal']]
            temporal_correct = result['pondere_temporal'] == metadata['true_temporal']
            
            temporal_status = "CORRECT" if temporal_correct else "INCORRECT"
            
            print("\nBarycentre pondéré confiance:")
            print("  Époque: {} ({})".format(temporal_pred, temporal_status))
            
            year_diff = abs(result['pondere_year'] - metadata['true_year'])
            year_accurate = year_diff <= 25
            year_status = "CORRECT" if year_accurate else "INCORRECT"
            
            print("  Année estimée: {:.1f} (diff: {:.1f} ans) ({})".format(
                result['pondere_year'], year_diff, year_status))
        
        return result
    
    def list_books(self, limit=10):
        print("\nLivres disponibles:")
        print("-" * 60)
        
        for i, book_id in enumerate(self.book_ids[:limit]):
            metadata = self.get_book_metadata(book_id)
            print("{}. {} - {} ({})".format(
                i+1, metadata['title'], metadata['author'], metadata['true_temporal_str']))
        
        if len(self.book_ids) > limit:
            print("... et {} autres livres".format(len(self.book_ids) - limit))
    
    def search_books(self, query):
        query = query.lower()
        results = []
        
        for book_id in self.book_ids:
            metadata = self.get_book_metadata(book_id)
            if query in metadata['title'].lower() or query in metadata['author'].lower():
                results.append((book_id, metadata))
        
        print("\nRésultats de recherche ({} trouvés):".format(len(results)))
        print("-" * 60)
        
        for i, (book_id, metadata) in enumerate(results):
            print("{}. {} - {} ({})".format(
                i+1, metadata['title'], metadata['author'], metadata['true_temporal_str']))
        
        return results
    
    def interactive_session(self):
        print("\nSession interactive")
        print("=" * 60)
        print("Commandes disponibles:")
        print("- 'random' : Tester un livre aléatoire")
        print("- 'list [N]' : Lister N livres (défaut: 10)")
        print("- 'search TERME' : Rechercher par titre ou auteur")
        print("- 'test ID' : Tester un livre spécifique")
        print("- 'help' : Afficher cette aide")
        print("- 'quit' : Quitter")
        
        while True:
            command = input("\n> ").strip()
            
            if command.lower() == "quit":
                print("Au revoir!")
                break
            
            elif command.lower() == "help":
                print("\nCommandes disponibles:")
                print("- 'random' : Tester un livre aléatoire")
                print("- 'list [N]' : Lister N livres (défaut: 10)")
                print("- 'search TERME' : Rechercher par titre ou auteur")
                print("- 'test ID' : Tester un livre spécifique")
                print("- 'quit' : Quitter")
            
            elif command.lower() == "random":
                self.test_random_book()
            
            elif command.lower().startswith("list"):
                parts = command.split()
                limit = 10
                if len(parts) > 1 and parts[1].isdigit():
                    limit = int(parts[1])
                self.list_books(limit)
            
            elif command.lower().startswith("search"):
                query = command[7:].strip()
                if not query:
                    print("Terme de recherche manquant")
                    continue
                self.search_books(query)
            
            elif command.lower().startswith("test"):
                book_id = command[5:].strip()
                if not book_id:
                    print("ID de livre manquant")
                    continue
                self.test_book(book_id)
            
            else:
                print("Commande inconnue. Tapez 'help' pour voir les commandes disponibles.")

def main():
    print("Interface de test sophistiquée BAM")
    print("=" * 60)
    
    # Charger les données de test
    test_data = load_test_data()
    if not test_data:
        print("Impossible de charger les données de test.")
        return
    
    # Charger le modèle
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("Impossible de charger le modèle. Vérifiez que l'entraînement a été effectué.")
        return
    
    # Créer et lancer l'interface
    tester = BookTester(model, tokenizer, test_data)
    tester.interactive_session()

if __name__ == "__main__":
    main()
