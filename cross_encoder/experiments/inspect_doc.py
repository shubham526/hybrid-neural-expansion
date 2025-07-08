# inspect_doc.py
import ir_datasets
import itertools

# The dataset you are using
DATASET_NAME = 'disks45/nocr/trec-robust-2004'

# The index where your script gets stuck. tqdm's count is 0-indexed.
PROBLEM_DOC_INDEX = 46485

print(f"Loading dataset: {DATASET_NAME}")
dataset = ir_datasets.load(DATASET_NAME)

print(f"Attempting to access document at index: {PROBLEM_DOC_INDEX}")

try:
    # Use itertools.islice to efficiently jump to the problematic document
    # We will inspect the one it stuck on and the next few, just in case.
    target_docs = itertools.islice(dataset.docs_iter(),
                                   PROBLEM_DOC_INDEX,
                                   PROBLEM_DOC_INDEX + 5)

    for i, doc in enumerate(target_docs):
        print("\n" + "=" * 80)
        print(f"Inspecting Document at Index: {PROBLEM_DOC_INDEX + i}")
        print("=" * 80)

        # Print all available attributes of the doc object
        # to see what we are dealing with.
        doc_id = getattr(doc, 'doc_id', 'N/A')
        print(f"  doc_id: {doc_id}")

        # Access the core text-like fields
        text = getattr(doc, 'text', 'N/A')
        body = getattr(doc, 'body', 'N/A')

        print(f"\n--- Raw 'text' attribute (first 500 chars) ---")
        print(str(text)[:500])

        print(f"\n--- Raw 'body' attribute (first 500 chars) ---")
        print(str(body)[:500])
        print("\n" + "=" * 80)


except Exception as e:
    print(f"\n🚨 An error occurred while trying to access the document at or after index {PROBLEM_DOC_INDEX}: {e}")