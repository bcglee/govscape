#!/bin/bash

# Usage: ./clear_s3_prod.sh s3://your-bucket/path/to/directory/

S3_PATH=s3://bcgl-public-bucket/prod-serving/*

# Remove all objects under the specified S3 path
s5cmd rm "$S3_PATH"
