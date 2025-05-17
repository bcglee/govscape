import os
import time
import multiprocessing
import pandas as pd
from functools import partial
from govscape import download_cdx_parquet, compute_single_cdx_counts, list_cdx_parquet_files


def process_single_file(file, local_dir="/tmp/govscape/"):
    """Process a single CDX file - this runs in a separate process"""
    try:
        # Download the file
        download_cdx_parquet("eotarchive", file, local_dir)
        local_filename = os.path.join(local_dir, os.path.basename(file))
        
        # Compute counts
        output = compute_single_cdx_counts(local_filename)
        
        # Clean up
        os.remove(local_filename)
        
        return output
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return pd.DataFrame(columns=["domain", "domain_count"])

def process_with_pool(cdx_files, num_processes=None):
    """Main processing function using process pool"""
    start = time.time()
    
    # Create a partial function with fixed local_dir parameter
    process_func = partial(process_single_file, local_dir="/tmp/govscape/")
    
    # Create process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process files in parallel
        results = pool.map(process_func, cdx_files)
    
    # Combine results
    final_output = pd.concat(results, ignore_index=True)
    
    # Aggregate counts
    final_output = final_output.groupby("domain")["domain_count"].sum().reset_index()
    
    # Sort and save results
    final_output = final_output.sort_values("domain_count", ascending=False)
    
    print(f"Processed {len(cdx_files)} files in {time.time()-start:.2f} seconds")
    print(final_output.head())
    
    final_output.to_csv("data/cdx_summary_results_pdfs.csv", index=False)
    
    return final_output

if __name__ == "__main__":
    cdx_files = list_cdx_parquet_files()
    final_output = process_with_pool(cdx_files, num_processes=8)

