import pytest
import shutil
import os

txt_directory = "test_files/text"
embed_directory = "test_files/embeddings"
image_directory = "test_files/images"

dirs_to_remove = [txt_directory, embed_directory, image_directory]

@pytest.fixture(scope="session", autouse=True)
def cleanup_directories():
    # Pre-Test Setup (nothing to be done)
    yield 
    # Post-Test Cleanup
    print("Cleaning up test directories...")
    for directory in dirs_to_remove:
        shutil.rmtree(directory, ignore_errors=True)
