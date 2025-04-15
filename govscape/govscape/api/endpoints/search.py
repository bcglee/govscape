from flask import request, current_app
from flask_restx import Namespace, Resource, fields

# Create namespace
ns = Namespace('search', description='Search operations')

# Define models
search_input = ns.model('SearchInput', {
    'query': fields.String(required=True, description='Search query text')
})

search_result = ns.model('SearchResult', {
    'pdf': fields.String(description='PDF file path'),
    'page': fields.String(description='Page number'),
    'distance': fields.Float(description='Distance score'),
    'jpeg': fields.String(description='JPEG image path')
})

search_response = ns.model('SearchResponse', {
    'results': fields.List(fields.Nested(search_result))
})

@ns.route('/')
class Search(Resource):
    @ns.doc('search_documents')
    @ns.expect(search_input, validate=True)
    @ns.response(200, 'Success', search_response)
    @ns.response(400, 'Invalid input')
    def post(self):
        """Search for documents matching the query"""
        data = request.get_json()
        if not data or 'query' not in data:
            return {"status": "error", "message": "Missing 'query' parameter"}, 400
        
        query = data['query']
        if not query.strip():
            return {"status": "error", "message": "Query cannot be empty"}, 400
        
        server = current_app.server
        
        return server.search(query)
