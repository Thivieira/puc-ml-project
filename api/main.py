# configuração para não exibir os warnings

from flask import Flask, send_from_directory
from flask_restx import Api, Resource, fields
import logging
from flask_cors import CORS
import warnings
import os
from models.spam_classifier import get_spam_classifier
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Habilitar CORS para o frontend

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Api(app, doc="/docs", title="SMS Spam Classifier API", 
          description="API para classificação de SMS como spam ou não spam usando machine learning", default_label="SMS Spam Classifier API")

# Get the spam classifier instance
spam_classifier = get_spam_classifier()

sms_input = api.model("SMS", {
    "text": fields.String(required=True, description="Texto da mensagem a ser classificada")
})

@api.route("/predict")
class Predict(Resource):
    @api.expect(sms_input)
    @api.response(200, "Sucesso", model=api.model("Result", {
        "spam": fields.Boolean,
        "prob": fields.Float
    }))
    @api.response(400, "Erro na requisição")
    def post(self):
        """
        Classifica uma mensagem de texto como spam ou não spam.
        
        IMPORTANTE: Por questões de privacidade e LGPD, as mensagens não são armazenadas.
        Para produção, recomenda-se implementar anonimização de dados pessoais.
        """
        try:
            texto = api.payload.get("text")
            
            if not texto or not texto.strip():
                return {"error": "Texto da mensagem é obrigatório"}, 400
            
            # Log apenas para debugging (sem armazenar o conteúdo)
            logger.info(f"Classificando mensagem de {len(texto)} caracteres")
            
            # Classificação usando o novo classificador
            result = spam_classifier.predict(texto)
            
            logger.info(f"Resultado: spam={result['spam']}, prob={result['prob']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Erro na classificação: {str(e)}")
            return {"error": "Erro interno na classificação"}, 500

def get_frontend_dist_dir():
    """Get the frontend dist directory path"""
    # In Docker, the frontend dist is mounted at /app/front/dist
    # In development, it's at ../front/dist relative to the API directory
    docker_path = "/app/front/dist"
    dev_path = os.path.join(os.path.dirname(__file__), "../front/dist")
    
    logger.info(f"Checking Docker path: {docker_path} (exists: {os.path.exists(docker_path)})")
    logger.info(f"Checking dev path: {dev_path} (exists: {os.path.exists(dev_path)})")
    
    if os.path.exists(docker_path):
        logger.info(f"Using Docker path: {docker_path}")
        return docker_path
    elif os.path.exists(dev_path):
        logger.info(f"Using dev path: {dev_path}")
        return dev_path
    else:
        logger.error(f"Neither path exists. Docker: {docker_path}, Dev: {dev_path}")
        raise FileNotFoundError(f"Frontend dist directory not found at {docker_path} or {dev_path}")

@app.route("/")
def home():
    """Serve the main HTML file from frontend dist"""
    try:
        dist_dir = get_frontend_dist_dir()
        logger.info(f"Serving index.html from: {dist_dir}")
        return send_from_directory(dist_dir, "index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return {"error": f"Frontend not available: {str(e)}"}, 500

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from frontend dist"""
    try:
        dist_dir = get_frontend_dist_dir()
        logger.info(f"Serving static file: {filename} from: {dist_dir}")
        return send_from_directory(dist_dir, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return {"error": f"Static file not available: {str(e)}"}, 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors by serving the main HTML file (for SPA routing)"""
    dist_dir = get_frontend_dist_dir()
    try:
        return send_from_directory(dist_dir, "index.html")
    except FileNotFoundError:
        return {"error": "Frontend not built. Please run 'npm run build' in the front directory."}, 404

if __name__ == "__main__":
    logger.info("Iniciando servidor SMS Spam Classifier API...")
    logger.info("Documentação disponível em: http://localhost:8000/docs")
    app.run(host="0.0.0.0", port=8000, debug=True)
