import os
import re
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class SpamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrator de features para classificação de spam em SMS.
    Baseado em características linguísticas e padrões comuns de spam.
    """
    
    def __init__(self):
        self.spam_keywords = [
            'free', 'winner', 'won', 'prize', 'claim', 'urgent', 'congratulations',
            'selected', 'iphone', 'computer', 'virus', 'antivirus', 'download',
            'bank', 'account', 'verify', 'suspended', 'bonus', 'money', 'cash',
            'limited', 'exclusive', 'guaranteed', 'risk-free', 'act now', 'call now',
            'text now', 'ringtone', 'viagra', 'lottery', 'credit', 'loan', 'debt',
            'sms', 'offer', 'discount', 'save'
        ]
        
        self.spam_patterns = [
            r'\b(?:FREE|FREE!)\b',
            r'\b(?:WINNER|WON)\b',
            r'\b(?:PRIZE|CLAIM)\b',
            r'\b(?:URGENT|URGENT!)\b',
            r'\b(?:CONGRATULATIONS!)\b',
            r'\b(?:SELECTED|YOU HAVE BEEN SELECTED)\b',
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
            feature_dict = {}
            feature_dict['exclamation_count'] = text_str.count('!')
            feature_dict['question_count'] = text_str.count('?')
            feature_dict['uppercase_count'] = sum(1 for c in text if c.isupper())
            feature_dict['digit_count'] = sum(1 for c in text if c.isdigit())
            feature_dict['text_length'] = len(text)
            feature_dict['word_count'] = len(text_str.split())
            feature_dict['spam_keyword_count'] = sum(1 for keyword in self.spam_keywords if keyword in text_str)
            feature_dict['spam_pattern_count'] = sum(1 for pattern in self.spam_patterns if re.search(pattern, text_upper))
            feature_dict['has_urgent'] = 1 if 'urgent' in text_str else 0
            feature_dict['has_free'] = 1 if 'free' in text_str else 0
            feature_dict['has_winner'] = 1 if 'winner' in text_str or 'won' in text_str else 0
            feature_dict['has_prize'] = 1 if 'prize' in text_str else 0
            feature_dict['has_claim'] = 1 if 'claim' in text_str else 0
            feature_dict['has_click'] = 1 if 'click' in text_str else 0
            feature_dict['has_congratulations'] = 1 if 'congratulations' in text_str else 0
            feature_dict['has_selected'] = 1 if 'selected' in text_str else 0
            feature_dict['has_iphone'] = 1 if 'iphone' in text_str else 0
            feature_dict['has_computer'] = 1 if 'computer' in text_str else 0
            feature_dict['has_virus'] = 1 if 'virus' in text_str else 0
            feature_dict['has_antivirus'] = 1 if 'antivirus' in text_str else 0
            feature_dict['has_download'] = 1 if 'download' in text_str else 0
            feature_dict['has_bank'] = 1 if 'bank' in text_str else 0
            feature_dict['has_account'] = 1 if 'account' in text_str else 0
            feature_dict['has_verify'] = 1 if 'verify' in text_str else 0
            feature_dict['has_suspended'] = 1 if 'suspended' in text_str else 0
            feature_dict['has_bonus'] = 1 if 'bonus' in text_str else 0
            feature_dict['has_money'] = 1 if 'money' in text_str else 0
            feature_dict['has_cash'] = 1 if 'cash' in text_str else 0
            feature_dict['has_limited'] = 1 if 'limited' in text_str else 0
            feature_dict['has_exclusive'] = 1 if 'exclusive' in text_str else 0
            feature_dict['has_guaranteed'] = 1 if 'guaranteed' in text_str else 0
            feature_dict['has_risk_free'] = 1 if 'risk-free' in text_str else 0
            feature_dict['has_act_now'] = 1 if 'act now' in text_str else 0
            feature_dict['has_call_now'] = 1 if 'call now' in text_str else 0
            feature_dict['has_text_now'] = 1 if 'text now' in text_str else 0
            feature_dict['has_ringtone'] = 1 if 'ringtone' in text_str else 0
            feature_dict['has_viagra'] = 1 if 'viagra' in text_str else 0
            feature_dict['has_lottery'] = 1 if 'lottery' in text_str else 0
            feature_dict['has_credit'] = 1 if 'credit' in text_str else 0
            feature_dict['has_loan'] = 1 if 'loan' in text_str else 0
            feature_dict['has_debt'] = 1 if 'debt' in text_str else 0
            feature_dict['has_sms'] = 1 if 'sms' in text_str else 0
            feature_dict['has_offer'] = 1 if 'offer' in text_str else 0
            feature_dict['has_discount'] = 1 if 'discount' in text_str else 0
            feature_dict['has_save'] = 1 if 'save' in text_str else 0
            
            # Proporções
            if len(text) > 0:
                feature_dict['uppercase_ratio'] = float(feature_dict['uppercase_count']) / len(text)
            else:
                feature_dict['uppercase_ratio'] = 0.0
            
            if feature_dict['word_count'] > 0:
                feature_dict['avg_word_length'] = float(feature_dict['text_length']) / feature_dict['word_count']
            else:
                feature_dict['avg_word_length'] = 0.0
            
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

class SpamClassifier:
    """
    Classificador de spam para SMS usando machine learning.
    """
    
    def __init__(self, model_path="ML/spam_model.joblib"):
        """
        Inicializa o classificador carregando o modelo treinado.
        
        Args:
            model_path (str): Caminho para o arquivo do modelo treinado
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo treinado do arquivo"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Modelo carregado com sucesso de: {self.model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict(self, text):
        """
        Classifica uma mensagem de texto como spam ou não spam.
        
        Args:
            text (str): Texto da mensagem a ser classificada
            
        Returns:
            dict: Dicionário com 'spam' (bool) e 'prob' (float)
        """
        if not text or not text.strip():
            raise ValueError("Texto da mensagem é obrigatório")
        
        try:
            # Classificação
            pred = self.model.predict([text])[0]
            prob = self.model.predict_proba([text])[0][1]
            
            result = {
                "spam": bool(pred),
                "prob": float(prob)
            }
            
            logger.info(f"Classificação: spam={result['spam']}, prob={result['prob']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Erro na classificação: {str(e)}")
            raise

# Instância global do classificador
spam_classifier = None

def get_spam_classifier():
    """
    Retorna a instância global do classificador de spam.
    Cria uma nova instância se não existir.
    
    Returns:
        SpamClassifier: Instância do classificador
    """
    global spam_classifier
    if spam_classifier is None:
        spam_classifier = SpamClassifier()
    return spam_classifier 