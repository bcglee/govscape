from flask_restx import Api

api = Api(
    version='1.0',
    title='GovScape API',
    description='A RESTful API for searching government PDF documents',
    doc='/docs'
)
