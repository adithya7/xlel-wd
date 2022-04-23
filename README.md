# Multilingual Event Linking to Wikidata

## Dataset

The XLEL-WD dataset is available in the huggingface datasets hub.

- [adithya7/xlel_wd_dictionary](https://huggingface.co/datasets/adithya7/xlel_wd_dictionary): a dictionary of events from Wikidata. This includes the title and descriptions for each event.
- [adithya7/xlel_wd](https://huggingface.co/datasets/adithya7/xlel_wd): mention references from Wikipedia and Wikinews.

### Loading Dictionary

```python
# pip install datasets
from datasets import load_dataset, get_dataset_config_names

# list all available configurations
dictionary_configs = get_dataset_config_names("adithya7/xlel_wd_dictionary")

"""
Default configuration: wikidata
Specify the language code to get language-specific dictionary (e.g. wikidata.fr)
"""

# load the Wikidata-based event dictionary
xlel_wd_dictionary = load_dataset("adithya7/xlel_wd_dictionary")
```

### Loading Mentions

```python
from datasets import load_dataset, get_dataset_config_names

# list all configurations
configs = get_dataset_config_names("adithya7/xlel_wd")

"""
Three main configurations:
    - wikipedia-zero-shot (splits: train, dev, test)
    - wikinews-zero-shot (splits: test)
    - wikinews-cross-domain (splits: test)

Each of the above configurations return mentions from all languages.
Specific language data can be accessed by specifying the language tag (e.g. wikinews-zero-shot.fr)
"""

# loading specific configuration
xlel_wd_dataset = load_dataset("adithya7/xlel_wd", "wikinews-zero-shot")
```

## Citation

```bib
@article{pratapa-etal-2022-multilingual,
  title = {Multilingual Event Linking to Wikidata},
  author = {Pratapa, Adithya and Gupta, Rishubh and Mitamura, Teruko},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2204.06535},
}
```
