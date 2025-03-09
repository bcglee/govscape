import pytest
import shutil
import os

txt_directory = "test_files/text"
embed_directory = "test_files/embeddings"

dirs_to_remove = [txt_directory, embed_directory]

@pytest.fixture(scope="session", autouse=True)
def cleanup_directories():
    yield
    print("Cleaning up test directories...")
    for directory in dirs_to_remove:
        shutil.rmtree(directory, ignore_errors=True)
