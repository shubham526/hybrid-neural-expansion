# diagnose_hang.py
import ir_datasets
import itertools
import time
import psutil
import os

# --- Configuration ---
DATASET_NAME = 'disks45/nocr/trec-robust-2004'
# We'll start just before the point where it usually hangs
START_INDEX = 46480
# How many documents to process after the start index
DOCS_TO_PROCESS = 100

# Get the current process for memory tracking
process = psutil.Process(os.getpid())

print("=" * 80)
print("Starting Final Diagnostic Script")
print(f"Loading dataset: {DATASET_NAME}")
print(f"Starting at index: {START_INDEX}")
print("This script has NO ColBERT code. It only tests the ir_datasets iterator.")
print("=" * 80)

try:
    dataset = ir_datasets.load(DATASET_NAME)

    # Create an iterator that starts near the problem area
    iterator = itertools.islice(dataset.docs_iter(), START_INDEX, START_INDEX + DOCS_TO_PROCESS)

    # Loop through the documents one by one
    for i, doc in enumerate(iterator):
        # 1. Get memory usage
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)  # Memory in MB

        # 2. Immediately try to access the doc_id
        doc_id = getattr(doc, 'doc_id', 'N/A')

        # 3. Print status
        print(f"Index: {START_INDEX + i}, Doc ID: {doc_id}, Memory: {mem_mb:.2f} MB")

        # A tiny sleep to make it possible to CTRL+C if needed
        time.sleep(0.01)

    print("\n✅ Diagnostic script completed successfully without hanging!")

except Exception as e:
    print(f"\n🚨 An error occurred during the diagnostic script: {e}")