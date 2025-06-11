import pytest
import shutil
import os

dirs_to_remove = ["test_files/small_test_data/text",
                  "test_files/small_test_data/embeddings",
                  "test_files/small_test_data/images",
                  "test_files/large_test_data/text",
                  "test_files/large_test_data/embeddings",
                  "test_files/large_test_data/images"]

@pytest.fixture(scope="session", autouse=True)
def cleanup_directories():
    # Pre-Test Setup (nothing to be done)
    yield 
    # Post-Test Cleanup
    print("Cleaning up test directories...")
    for directory in dirs_to_remove:
        shutil.rmtree(directory, ignore_errors=True)

