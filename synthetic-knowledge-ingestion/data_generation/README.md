
## DATA Generation
Data generation is an important piece of the SKI framework. It is able to generate different formats of data that can be used for RAG, SFT and CPT.

To generate data, modify the configuration in `data_generator.py` and then run `python data_generator.py`.

If the task data is not already downloaded, it will download the data using [Beir](https://github.com/beir-cellar/beir/tree/main). If a task is not downloadble like bioasq, you would need to manullay configure it.

All modes in `ski_mode.py` are supported. When generating for training purpose, set `for_training=True` and ensure your dataset has training split available.
