from flask import Flask # type: ignore
from flask_cors import CORS # type: ignore
from .routes import api # type: ignore

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api, url_prefix='/vitals')
    return app
