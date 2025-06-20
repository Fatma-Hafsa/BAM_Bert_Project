

"""
Interface de test interactive pour le projet BAM
Permet de tester le modèle sur de nouveaux textes en temps réel
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CamembertTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bam_project import AdvancedBertBAM, Config

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print(" Chargement du modèle BAM...")
    
    model_path = os.path.join(config.MODELS_DIR, 'bam_student_best.pth')
    
    if not os.path.exists(model_path):
        print(f" Modèle non trouvé: {model_path}")
        return None, None
    
    try:
        # Charger le modèle
        model = AdvancedBertBAM(config.BERT_MODEL)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Charger le tokenizer
        tokenizer = CamembertTokenizer.from_pretrained(config.BERT_MODEL)
        
        print("Modèle chargé avec succès!")
        return model, tokenizer
        
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        return None, None

def predict_single_text(model, tokenizer, text, max_length=512):
    model.eval()
    
    # Tokenisation
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
        predictions = {}
        
        # Prédiction genre
        if 'gender_logits' in outputs:
            gender_probs = F.softmax(outputs['gender_logits'], dim=1).cpu().numpy()[0]
            gender_pred = np.argmax(gender_probs)
            predictions['gender'] = {
                'prediction': gender_pred,
                'probabilities': gender_probs,
                'confidence': float(gender_probs[gender_pred])
            }
        
        # Prédiction temporelle
        if 'temporal_logits' in outputs:
            temporal_probs = F.softmax(outputs['temporal_logits'], dim=1).cpu().numpy()[0]
            temporal_pred = np.argmax(temporal_probs)
            
            # Calcul du barycentre pondéré (méthode softmax complet)
            temporal_barycentre = 0.0
            for i, prob in enumerate(temporal_probs):
                temporal_barycentre += config.TEMPORAL_MIDPOINTS[i] * prob
            
            # Calcul du barycentre pondéré par confiance (seulement classe prédite)
            temporal_barycentre_confidence = config.TEMPORAL_MIDPOINTS[temporal_pred]
            
            predictions['temporal'] = {
                'prediction': temporal_pred,
                'probabilities': temporal_probs,
                'confidence': float(temporal_probs[temporal_pred]),
                'barycentre_year': temporal_barycentre,
                'barycentre_confidence_year': temporal_barycentre_confidence,
                'predicted_period': config.TEMPORAL_PERIODS[temporal_pred]
            }
    
    return predictions

def display_predictions(predictions):
    print("\n PRÉDICTIONS:")
    print("-" * 40)
    
    # Genre
    if 'gender' in predictions:
        gender_pred = predictions['gender']['prediction']
        gender_conf = predictions['gender']['confidence']
        gender_label = "Homme" if gender_pred == 1 else "Femme"
        
        print(f" SEXE DE L'AUTEUR:")
        print(f"   Prédiction: {gender_label}")
        print(f"   Confiance: {gender_conf:.1%}")
        
        # Distribution des probabilités
        probs = predictions['gender']['probabilities']
        print(f"   Détail: Femme {probs[0]:.1%} | Homme {probs[1]:.1%}")
    
    # Période temporelle
    if 'temporal' in predictions:
        temporal_pred = predictions['temporal']['prediction']
        temporal_conf = predictions['temporal']['confidence']
        period = predictions['temporal']['predicted_period']
        barycentre = predictions['temporal']['barycentre_year']
        
        print(f"\n PÉRIODE TEMPORELLE:")
        print(f"   Prédiction: {period}")
        print(f"   Confiance: {temporal_conf:.1%}")
        print(f"   Année estimée (barycentre softmax): {barycentre:.0f}")
        
        # Top 3 des périodes les plus probables
        probs = predictions['temporal']['probabilities']
        top_periods = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"   Top 3 périodes:")
        for i, (period_idx, prob) in enumerate(top_periods, 1):
            period_name = config.TEMPORAL_PERIODS[period_idx]
            print(f"     {i}. {period_name}: {prob:.1%}")

def interactive_text_test(model, tokenizer):
    print("\n INTERFACE DE TEST INTERACTIF")
    print("="*60)
    print("Testez le modèle BAM sur vos propres textes !")
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    
    while True:
        print("\n" + "-"*50)
        user_input = input(" Entrez votre texte (ou 'quit'/'help'): ").strip()
        
        if user_input.lower() == 'quit':
            print(" Au revoir !")
            break
        elif user_input.lower() == 'help':
            print("\n AIDE:")
            print("- Entrez un texte littéraire français")
            print("- Le modèle prédit le sexe de l'auteur et la période")
            print("- Exemples de commandes:")
            print("  'exemple' - Teste avec des exemples prédéfinis")
            print("  'fichier' - Charge un texte depuis un fichier")
            print("  'quit' - Quitter")
            continue
        elif user_input.lower() == 'exemple':
            # Exemples prédéfinis
            exemples = [
                {
                    'text': "Il était une fois, dans un château lointain, une princesse qui rêvait de liberté. Les murs dorés de sa prison dorée ne pouvaient contenir son esprit aventureux.",
                    'description': "Style conte classique"
                },
                {
                    'text': "Le métro parisien grondait sous nos pieds. Dans cette foule anonyme, chacun portait ses secrets, ses espoirs brisés, ses amours perdues.",
                    'description': "Style moderne urbain"
                },
                {
                    'text': "Mon cher ami, permettez-moi de vous conter cette histoire extraordinaire qui bouleversa ma tranquille existence bourgeoise.",
                    'description': "Style XIXe siècle"
                },
                {
                    'text': "Elle regardait par la fenêtre de sa chambre, songeant aux lettres qu'elle n'oserait jamais écrire. Son cœur battait la chamade à la pensée de cet amour impossible.",
                    'description': "Style romantique"
                }
            ]
            
            for i, exemple in enumerate(exemples, 1):
                print(f"\n Exemple {i}: {exemple['description']}")
                print(f"Texte: {exemple['text'][:100]}...")
                predictions = predict_single_text(model, tokenizer, exemple['text'])
                display_predictions(predictions)
            continue
        elif user_input.lower() == 'fichier':
            file_path = input(" Chemin du fichier texte: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read().strip()
                if file_text:
                    print(f" Fichier chargé ({len(file_text)} caractères)")
                    if len(file_text) > 2000:
                        file_text = file_text[:2000]
                        print(" Texte tronqué à 2000 caractères")
                    predictions = predict_single_text(model, tokenizer, file_text)
                    display_predictions(predictions)
                else:
                    print(" Fichier vide")
            except Exception as e:
                print(f"Erreur lecture fichier: {e}")
            continue
        
        if not user_input:
            print(" Texte vide, veuillez entrer du texte")
            continue
        
        try:
            predictions = predict_single_text(model, tokenizer, user_input)
            display_predictions(predictions)
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")

def main():
    print("INTERFACE DE TEST BAM")
    print("="*60)
    
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print(" Impossible de charger le modèle. Vérifiez que l'entraînement a été effectué.")
        return
    
    interactive_text_test(model, tokenizer)

if __name__ == "__main__":
    main()

