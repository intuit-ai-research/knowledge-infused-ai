## Rag & Retrieval 
For Retreival, four tasks are run: `bioasq`, `hotpotqa`, `fiqa` and `nq`. For Rag, three tasks are run: `bioasq`, `hotpotqa` and `nq`.

To get the retreival data for bioasq, follow instructions [here](https://github.com/beir-cellar/beir/tree/main/examples/dataset#2-bioasq)

Available modes for retreival are: `SKIMode.Q`, `SKIMode.QC`, `SKIMode.QCASM`

To run experiments, first configure in `rag.py` and then run `python rag.py`
