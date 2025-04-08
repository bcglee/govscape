from flask import request
from flask_restx import Resource, fields
from ..api import api

# Define models directly in the endpoint
search_input = api.model('SearchInput', {
    'query': fields.String(required=True, description='Search query text')
})

search_result = api.model('SearchResult', {
    'pdf': fields.String(description='PDF file path'),
    'page': fields.String(description='Page number'),
    'distance': fields.Float(description='Distance score'),
    'jpeg': fields.String(description='JPEG image path')
})

search_response = api.model('SearchResponse', {
    'results': fields.List(fields.Nested(search_result))
})

class Search(Resource):
    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)
        self.context = self.api.app.context

    @api.doc('search_documents')
    @api.expect(search_input, validate=True)
    @api.response(200, 'Success', search_response)
    @api.response(400, 'Invalid input')
    def post(self):
        """Search for documents matching the query"""
        data = request.get_json()
        if not data or 'query' not in data:
            return {"status": "error", "message": "Missing 'query' parameter"}, 400
        
        query = data['query']
        if not query.strip():
            return {"status": "error", "message": "Query cannot be empty"}, 400
        
        # Process query
        query_embedding = self.context.model.text_to_embeddings(query)
        D, I = self.context.faiss_index.search(query_embedding, self.context.k)
        
        search_results = []
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                # parse file information for page
                pdf_name, _, page = self.context.npy_files[I[i][j]].rpartition('_')
                page, _, _ = page.rpartition('.')
                # create jpeg name
                jpeg = self.context.image_directory + "/" + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page + '.jpg'
                
                # add results to list
                search_results.append({
                    "pdf": pdf_name, 
                    "page": page, 
                    "distance": float(D[i][j]), 
                    "jpeg": jpeg
                })
        
        return {"results": search_results}
