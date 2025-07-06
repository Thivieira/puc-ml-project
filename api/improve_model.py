#!/usr/bin/env python3
"""
Script para melhorar o modelo de classificação de spam
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """
    Pré-processamento mais robusto do texto
    """
    if pd.isna(text):
        return ""
    
    # Converter para string
    text = str(text).lower()
    
    # Remover URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remover números de telefone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
    
    # Remover caracteres especiais mas manter alguns importantes
    text = re.sub(r'[^\w\s!?$%#@*&]', ' ', text)
    
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_features(text):
    """
    Extrair features específicas para spam
    """
    features = {}
    
    # Contagem de caracteres especiais
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_count'] = sum(1 for c in text if c.isupper())
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    
    # Palavras-chave de spam
    spam_keywords = [
        'urgent', 'free', 'winner', 'won', 'prize', 'claim', 'click', 'limited',
        'offer', 'discount', 'save', 'money', 'cash', 'bonus', 'congratulations',
        'selected', 'exclusive', 'guaranteed', 'risk-free', 'act now', 'call now',
        'text', 'sms', 'ringtone', 'viagra', 'lottery', 'credit', 'loan', 'debt',
        'bank', 'account', 'verify', 'suspended', 'virus', 'antivirus', 'download'
    ]
    
    text_lower = text.lower()
    features['spam_keyword_count'] = sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    # Comprimento do texto
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Proporção de maiúsculas
    if len(text) > 0:
        features['uppercase_ratio'] = features['uppercase_count'] / len(text)
    else:
        features['uppercase_ratio'] = 0
    
    return features

def create_improved_pipeline():
    """
    Criar pipeline melhorado com múltiplos classificadores
    """
    
    # TF-IDF com parâmetros otimizados
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),  # Unigramas, bigramas e trigramas
        min_df=2,
        max_df=0.95,
        stop_words='english',
        strip_accents='unicode'
    )
    
    # Múltiplos classificadores
    svm = SVC(probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    nb = MultinomialNB()
    
    # Pipeline principal com TF-IDF + SVM
    main_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', svm)
    ])
    
    return main_pipeline, [svm, rf, nb]

def train_improved_model():
    """
    Treinar modelo melhorado
    """
    print("🔄 Treinando modelo melhorado...")
    print("=" * 50)
    
    # Carregar dados
    df = pd.read_csv("ML/SMSSpamCollection", sep="\t", header=None, names=['label', 'text'])
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset: {len(df)} mensagens ({df['target'].sum()} spam, {len(df)-df['target'].sum()} ham)")
    
    # Pré-processar texto
    print("🧹 Pré-processando texto...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['target'], 
        test_size=0.2, random_state=42, stratify=df['target']
    )
    
    # Criar pipeline
    main_pipeline, classifiers = create_improved_pipeline()
    
    # Treinar modelo principal
    print("🎯 Treinando classificador principal (SVM)...")
    main_pipeline.fit(X_train, y_train)
    
    # Avaliar modelo principal
    y_pred = main_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy do modelo principal: {accuracy:.4f}")
    
    # Otimizar hiperparâmetros
    print("🔧 Otimizando hiperparâmetros...")
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear'],
        'tfidf__max_features': [3000, 5000, 7000]
    }
    
    grid_search = GridSearchCV(
        main_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor F1-Score: {grid_search.best_score_:.4f}")
    
    # Modelo final otimizado
    best_model = grid_search.best_estimator_
    
    # Avaliação final
    y_pred_final = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    
    print(f"\n📊 Resultados Finais:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print("\n" + classification_report(y_test, y_pred_final))
    
    # Salvar modelo melhorado
    joblib.dump(best_model, 'ML/improved_spam_model.joblib')
    print("\n💾 Modelo melhorado salvo como 'improved_spam_model.joblib'")
    
    return best_model, X_test, y_test

def test_improved_model():
    """
    Testar o modelo melhorado
    """
    print("\n🧪 Testando modelo melhorado...")
    print("=" * 50)
    
    # Carregar modelo melhorado
    model = joblib.load('ML/improved_spam_model.joblib')
    
    # Mensagens problemáticas identificadas anteriormente
    problem_messages = [
        "CONGRATULATIONS! You've been selected for a free iPhone!",
        "URGENT: Your computer has a virus! Download antivirus now!",
        "Hi, how are you? Let's meet for coffee tomorrow.",
        "URGENT! You have won a prize! Click here to claim!"
    ]
    
    expected = [1, 1, 0, 1]  # 1=spam, 0=ham
    
    print("📝 Testando mensagens problemáticas:")
    print("-" * 40)
    
    for i, (msg, exp) in enumerate(zip(problem_messages, expected), 1):
        # Pré-processar
        processed_msg = preprocess_text(msg)
        
        # Predição
        pred = model.predict([processed_msg])[0]
        prob = model.predict_proba([processed_msg])[0][1]
        
        result = "SPAM" if pred == 1 else "HAM"
        expected_text = "SPAM" if exp == 1 else "HAM"
        status = "✅" if pred == exp else "❌"
        
        print(f"{i}. {status} {result:4s} (prob: {prob:.3f}) - {expected_text:4s}")
        print(f"    \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print()

if __name__ == "__main__":
    # Treinar modelo melhorado
    best_model, X_test, y_test = train_improved_model()
    
    # Testar modelo melhorado
    test_improved_model()
    
    print("\n🎉 Modelo melhorado pronto!")
    print("Para usar, substitua 'spam_model.joblib' por 'improved_spam_model.joblib' no main.py") 