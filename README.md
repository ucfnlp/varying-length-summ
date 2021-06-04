# A New Approach to Overgenerating and Scoring Abstractive Summaries
We provide the source code for the paper  **"[A New Approach to Overgenerating and Scoring Abstractive Summaries](https://www.aclweb.org/anthology/2021.naacl-main.110.pdf)"** accepted at NAACL'21. If you find the code useful, please cite the following paper.

    @inproceedings{song2021new, 
        title={A New Approach to Overgenerating and Scoring Abstractive Summaries},
        author={Song, Kaiqiang and Wang, Bingqing and Feng, Zhe and Liu, Fei},
        booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
        pages={1392--1404},
        year={2021}
    }

## Dependencies

The code is written in Python (v3.7) and Pytorch (v1.7+). We suggest the following enviorment:

* A Linux machine (Ubuntu) with GPU
* [Python (v3.7+)](https://www.anaconda.com/download/)
* [Pytorch (v1.7+)](https://pytorch.org/)
* [Pyrouge](https://pypi.org/project/pyrouge/)
* [transformers (v2.3.0)](https://github.com/huggingface/transformers)

HINT: Since huggingface transformers is alternating very fast, you may need to modify a lot of stuff if you want to use a new version. Contact me if you get any trouble on it.

To install pyrouge and transformers, run the command below:

```
$ pip install pyrouge transformers==2.3.0
```

## For generating summaries with varying length

Step 1: clone this repo. Download trained [Our model](), move it to the working folder and uncompress it.

```
$ git clone https://github.com/ucfnlp/varying-length-summ.git
$ mv models.zip varying-length-summ
$ cd varying-length-summ
$ unzip models.zip
```

Step 2: Generating summaries with varying length from a raw input file. (please provide a standard reference) 
```
$ python run.py --do_test --parallel --input data/input.txt --standard data/reference.txt
```

HINT: you can modify the code in [run.py]() if you don't have any reference.


## For Selecting summaries with best quality

Step 1: Follow the prvious section about generating summaries with multiple length.

Step 2: Working on it


## For Downloading Summaries and Labels of different lengths on 500 sampled instances on gigaword

Working on it