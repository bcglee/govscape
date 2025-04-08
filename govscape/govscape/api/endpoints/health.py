from flask_restx import Resource, fields
from ..api import api

# Define health model directly in the endpoint
health_response = api.model('HealthResponse', {
    'status': fields.String(description='Server health status'),
    'embeddings_count': fields.Integer(description='Number of embeddings loaded')
})

class HealthCheck(Resource):
    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)
        self.context = self.api.app.context

    @api.doc('get_health')
    @api.response(200, 'Success', health_response)
    def get(self):
        """Get server health status"""
        return {
            "status": "healthy", 
            "embeddings_count": self.context.faiss_index.ntotal
        }
