# Local Research Papers

Place PDF or TXT documents in this folder before ingestion.

The actual research papers used during local testing are not committed because they may be third-party copyrighted material. Keeping this folder empty in Git also keeps the repository safe if it is later made public.

Recommended local flow:

```bat
copy path\to\papers\*.pdf papers\
python ingest.py --reset
```
