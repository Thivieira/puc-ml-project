#!/usr/bin/env python3
"""
Criando modelo robusto com features específicas para spam
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

class SpamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrator de features específicas para spam
    """
    
    def __init__(self):
        self.spam_keywords = [
            'urgent', 'free', 'winner', 'won', 'prize', 'claim', 'click', 'limited',
            'offer', 'discount', 'save', 'money', 'cash', 'bonus', 'congratulations',
            'selected', 'exclusive', 'guaranteed', 'risk-free', 'act now', 'call now',
            'text', 'sms', 'ringtone', 'viagra', 'lottery', 'credit', 'loan', 'debt',
            'bank', 'account', 'verify', 'suspended', 'virus', 'antivirus', 'download',
            'iphone', 'iphone!', 'computer', 'mobile', 'number', 'awarded', 'bonus',
            'claim now', 'click here', 'limited time', 'exclusive offer'
        ]
        
        self.spam_patterns = [
            r'\b(?:URGENT|FREE|WINNER|PRIZE|CLAIM|CLICK|LIMITED|OFFER|BONUS|CONGRATULATIONS)\b',
            r'\b(?:ACT NOW|CALL NOW|TEXT NOW|DOWNLOAD NOW)\b',
            r'\b(?:YOUR (?:ACCOUNT|BANK|COMPUTER|MOBILE|NUMBER))\b',
            r'\b(?:HAS BEEN (?:SUSPENDED|AWARDED|SELECTED))\b',
            r'\b(?:NEEDS (?:VERIFICATION|DOWNLOAD|ANTIVIRUS))\b',
            r'\b(?:FREE (?:RINGTONE|VIAGRA|CREDIT|LOTTERY|IPHONE))\b',
            r'\b(?:WIN (?:PRIZE|MONEY|CASH|BONUS))\b',
            r'\b(?:CONGRATULATIONS! YOU\'VE)\b',
            r'\b(?:URGENT: YOUR)\b',
            r'\b(?:CLICK HERE TO)\b'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            text_str = str(text).lower()
            text_upper = str(text).upper()
            
            # Features básicas
            feature_dict = {
                'exclamation_count': text_str.count('!'),
                'question_count': text_str.count('?'),
                'uppercase_count': sum(1 for c in text if c.isupper()),
                'digit_count': sum(1 for c in text if c.isdigit()),
                'text_length': len(text),
                'word_count': len(text_str.split()),
                'spam_keyword_count': sum(1 for keyword in self.spam_keywords if keyword in text_str),
                'spam_pattern_count': sum(1 for pattern in self.spam_patterns if re.search(pattern, text_upper)),
                'has_urgent': 1 if 'urgent' in text_str else 0,
                'has_free': 1 if 'free' in text_str else 0,
                'has_winner': 1 if 'winner' in text_str or 'won' in text_str else 0,
                'has_prize': 1 if 'prize' in text_str else 0,
                'has_claim': 1 if 'claim' in text_str else 0,
                'has_click': 1 if 'click' in text_str else 0,
                'has_congratulations': 1 if 'congratulations' in text_str else 0,
                'has_selected': 1 if 'selected' in text_str else 0,
                'has_iphone': 1 if 'iphone' in text_str else 0,
                'has_computer': 1 if 'computer' in text_str else 0,
                'has_virus': 1 if 'virus' in text_str else 0,
                'has_antivirus': 1 if 'antivirus' in text_str else 0,
                'has_download': 1 if 'download' in text_str else 0,
                'has_bank': 1 if 'bank' in text_str else 0,
                'has_account': 1 if 'account' in text_str else 0,
                'has_verify': 1 if 'verify' in text_str else 0,
                'has_suspended': 1 if 'suspended' in text_str else 0,
                'has_bonus': 1 if 'bonus' in text_str else 0,
                'has_money': 1 if 'money' in text_str else 0,
                'has_cash': 1 if 'cash' in text_str else 0,
                'has_limited': 1 if 'limited' in text_str else 0,
                'has_exclusive': 1 if 'exclusive' in text_str else 0,
                'has_guaranteed': 1 if 'guaranteed' in text_str else 0,
                'has_risk_free': 1 if 'risk-free' in text_str else 0,
                'has_act_now': 1 if 'act now' in text_str else 0,
                'has_call_now': 1 if 'call now' in text_str else 0,
                'has_text_now': 1 if 'text now' in text_str else 0,
                'has_ringtone': 1 if 'ringtone' in text_str else 0,
                'has_viagra': 1 if 'viagra' in text_str else 0,
                'has_lottery': 1 if 'lottery' in text_str else 0,
                'has_credit': 1 if 'credit' in text_str else 0,
                'has_loan': 1 if 'loan' in text_str else 0,
                'has_debt': 1 if 'debt' in text_str else 0,
                'has_sms': 1 if 'sms' in text_str else 0,
                'has_offer': 1 if 'offer' in text_str else 0,
                'has_discount': 1 if 'discount' in text_str else 0,
                'has_save': 1 if 'save' in text_str else 0,
            }
            
            # Proporções
            if len(text) > 0:
                feature_dict['uppercase_ratio'] = feature_dict['uppercase_count'] / len(text)
            else:
                feature_dict['uppercase_ratio'] = 0
            
            if feature_dict['word_count'] > 0:
                feature_dict['avg_word_length'] = feature_dict['text_length'] / feature_dict['word_count']
            else:
                feature_dict['avg_word_length'] = 0
            
            # Features compostas
            feature_dict['spam_score'] = (
                feature_dict['spam_keyword_count'] * 2 +
                feature_dict['spam_pattern_count'] * 3 +
                feature_dict['exclamation_count'] * 0.5 +
                feature_dict['uppercase_ratio'] * 10 +
                feature_dict['has_urgent'] * 5 +
                feature_dict['has_free'] * 3 +
                feature_dict['has_winner'] * 4 +
                feature_dict['has_prize'] * 4 +
                feature_dict['has_claim'] * 3 +
                feature_dict['has_congratulations'] * 4 +
                feature_dict['has_selected'] * 3 +
                feature_dict['has_iphone'] * 3 +
                feature_dict['has_computer'] * 2 +
                feature_dict['has_virus'] * 3 +
                feature_dict['has_antivirus'] * 3 +
                feature_dict['has_download'] * 2 +
                feature_dict['has_bank'] * 3 +
                feature_dict['has_account'] * 2 +
                feature_dict['has_verify'] * 3 +
                feature_dict['has_suspended'] * 4 +
                feature_dict['has_bonus'] * 3 +
                feature_dict['has_money'] * 2 +
                feature_dict['has_cash'] * 2 +
                feature_dict['has_limited'] * 2 +
                feature_dict['has_exclusive'] * 2 +
                feature_dict['has_guaranteed'] * 2 +
                feature_dict['has_risk_free'] * 3 +
                feature_dict['has_act_now'] * 4 +
                feature_dict['has_call_now'] * 3 +
                feature_dict['has_text_now'] * 3 +
                feature_dict['has_ringtone'] * 3 +
                feature_dict['has_viagra'] * 5 +
                feature_dict['has_lottery'] * 4 +
                feature_dict['has_credit'] * 2 +
                feature_dict['has_loan'] * 2 +
                feature_dict['has_debt'] * 2 +
                feature_dict['has_sms'] * 2 +
                feature_dict['has_offer'] * 2 +
                feature_dict['has_discount'] * 2 +
                feature_dict['has_save'] * 1
            )
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)

def preprocess_text(text):
    """
    Pré-processamento robusto do texto
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remover URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remover números de telefone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
    
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_robust_pipeline():
    """
    Criar pipeline robusto com features específicas
    """
    
    # TF-IDF para features de texto
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Features específicas para spam
    spam_features = SpamFeatureExtractor()
    
    # Combinar features
    feature_union = FeatureUnion([
        ('tfidf', tfidf),
        ('spam_features', spam_features)
    ])
    
    # Classificador
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Pipeline completo
    pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_robust_model():
    """
    Treinar modelo robusto
    """
    print("🛡️ Treinando modelo robusto...")
    print("=" * 50)
    
    # Carregar dados
    df = pd.read_csv("ML/SMSSpamCollection", sep="\t", header=None, names=['label', 'text'])
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset: {len(df)} mensagens ({df['target'].sum()} spam, {len(df)-df['target'].sum()} ham)")
    
    # Pré-processar
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['target'], 
        test_size=0.2, random_state=42, stratify=df['target']
    )
    
    # Criar pipeline
    pipeline = create_robust_pipeline()
    
    # Treinar
    print("🎯 Treinando modelo...")
    pipeline.fit(X_train, y_train)
    
    # Avaliar
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\n" + classification_report(y_test, y_pred))
    
    # Salvar modelo
    joblib.dump(pipeline, 'ML/robust_spam_model.joblib')
    print("\n💾 Modelo robusto salvo como 'robust_spam_model.joblib'")
    
    return pipeline, X_test, y_test

def test_robust_model():
    """
    Testar modelo robusto
    """
    print("\n🧪 Testando modelo robusto...")
    print("=" * 50)
    
    model = joblib.load('ML/robust_spam_model.joblib')
    
    # Mensagens problemáticas
    problem_messages = [
        "CONGRATULATIONS! You've been selected for a free iPhone!",
        "URGENT: Your computer has a virus! Download antivirus now!",
        "Hi, how are you? Let's meet for coffee tomorrow.",
        "URGENT! You have won a prize! Click here to claim!",
        "FREE RINGTONE text FIRST to 87131 for a poly",
        "Ok, I'll call you later",
        "Thanks for your help yesterday"
    ]
    
    expected = [1, 1, 0, 1, 1, 0, 0]  # 1=spam, 0=ham
    
    print("📝 Testando mensagens:")
    print("-" * 40)
    
    correct = 0
    for i, (msg, exp) in enumerate(zip(problem_messages, expected), 1):
        # Pré-processar
        processed_msg = preprocess_text(msg)
        
        # Predição
        pred = model.predict([processed_msg])[0]
        prob = model.predict_proba([processed_msg])[0][1]
        
        result = "SPAM" if pred == 1 else "HAM"
        expected_text = "SPAM" if exp == 1 else "HAM"
        status = "✅" if pred == exp else "❌"
        
        if pred == exp:
            correct += 1
        
        print(f"{i}. {status} {result:4s} (prob: {prob:.3f}) - {expected_text:4s}")
        print(f"    \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print()
    
    accuracy = correct / len(problem_messages)
    print(f"📊 Resultado: {correct}/{len(problem_messages)} corretos ({accuracy:.1%})")
    
    if accuracy >= 0.9:
        print("🎉 Modelo robusto funcionando muito bem!")
    elif accuracy >= 0.8:
        print("✅ Modelo robusto funcionando bem!")
    else:
        print("⚠️  Modelo ainda precisa de ajustes")

if __name__ == "__main__":
    # Treinar modelo robusto
    model, X_test, y_test = train_robust_model()
    
    # Testar modelo robusto
    test_robust_model()
    
    print("\n🎯 Para usar o modelo robusto:")
    print("1. Substitua 'spam_model.joblib' por 'robust_spam_model.joblib' no main.py")
    print("2. Reinicie o servidor Flask") 