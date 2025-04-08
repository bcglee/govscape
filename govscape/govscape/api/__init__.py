from .api import api
from .endpoints.health import HealthCheck
from .endpoints.search import Search

def init_api(app):
    """Initialize the Flask-RESTX API"""
    # Initialize the api with the app
    api.init_app(app)
    
    # Create namespaces
    health_ns = api.namespace('health', description='Health check operations')
    search_ns = api.namespace('search', description='Search operations')
    
    # Register endpoints
    health_ns.add_resource(HealthCheck, '')
    search_ns.add_resource(Search, '')
    
    return api
