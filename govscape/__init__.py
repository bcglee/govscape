# AI modified: 2026-06-28 00:00:00 efa6046e31518ce0c8f83630892d98a3b4a437bb
from importlib import import_module

__all__ = [
    "BGESmall_TextEmbeddingModel",
    "BGE_TextEmbeddingModel",
    "CLIP_VisualEmbeddingModel",
    "DataModel",
    "DiskANNIndex",
    "Dummy_TextEmbeddingModel",
    "Dummy_VisualEmbeddingModel",
    "FAISSIndex",
    "LanceDBKeywordIndex",
    "LuceneKeywordIndex",
    "PDFProcessingPipeline",
    "RemoteDirectoryIterator",
    "SQLiteKeywordIndex",
    "SQLiteMetadataIndex",
    "ST_TextEmbeddingModel",
    "Server",
    "ServerConfig",
    "WhooshKeywordIndex",
    "base_argument_parser",
    "build_data_loader",
    "extract_subdomain",
    "read_txt_file",
    "str2bool",
]

_EXPORTS = {
    "BGESmall_TextEmbeddingModel": (".text_embedding_models", "BGESmall_TextEmbeddingModel"),
    "BGE_TextEmbeddingModel": (".text_embedding_models", "BGE_TextEmbeddingModel"),
    "CLIP_VisualEmbeddingModel": (".visual_embedding_models", "CLIP_VisualEmbeddingModel"),
    "DataModel": (".config", "DataModel"),
    "Dummy_TextEmbeddingModel": (".text_embedding_models", "Dummy_TextEmbeddingModel"),
    "Dummy_VisualEmbeddingModel": (".visual_embedding_models", "Dummy_VisualEmbeddingModel"),
    "FAISSIndex": (".indexing", "FAISSIndex"),
    "LanceDBKeywordIndex": (".indexing", "LanceDBKeywordIndex"),
    "LuceneKeywordIndex": (".indexing", "LuceneKeywordIndex"),
    "PDFProcessingPipeline": (".pdf_processing_pipeline", "PDFProcessingPipeline"),
    "RemoteDirectoryIterator": (".data_loader", "RemoteDirectoryIterator"),
    "SQLiteKeywordIndex": (".indexing", "SQLiteKeywordIndex"),
    "SQLiteMetadataIndex": (".indexing", "SQLiteMetadataIndex"),
    "ST_TextEmbeddingModel": (".text_embedding_models", "ST_TextEmbeddingModel"),
    "Server": (".server", "Server"),
    "ServerConfig": (".config", "ServerConfig"),
    "WhooshKeywordIndex": (".indexing", "WhooshKeywordIndex"),
    "base_argument_parser": (".utils", "base_argument_parser"),
    "build_data_loader": (".data_loader", "build_data_loader"),
    "extract_subdomain": (".utils", "extract_subdomain"),
    "read_txt_file": (".utils", "read_txt_file"),
    "str2bool": (".utils", "str2bool"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
