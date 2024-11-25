# Gitor_plus

We propose *Gitor_plus* to capture the underlying connections among different code samples.  Specifically, given a source code database, we first tokenize all code samples to extract the pre-defined individual information (i.e., *keywords* and *code metrics*).  After obtaining all samples’ individual information, we leverage them to build a large *global sample graph* where each node is a code sample or a type of individual information.  Then we apply a node embedding technique on the global sample graph to extract all the samples’ vector representations.  After collecting all code samples’ vectors, we can either directly compute the cosine similarity between any two samples to quickly detect potential clone pairs or input these vectors into a neural network classifier to identify code clones.

The Gitor_plus mainly consists of six phases:

- `extract.py`: Used to extract global information (Keywords) from code samples.
- `metrics.py`: Used to extract code metrics from code samples.
- `train.py`: Used for node embedding.
- `train_fcl.py`: Used for Gitor’s neural network-based variants 
- `detect.py`: Used for code clone detection.
- `train_sca.py`, `train_sca_fcl.py`, `detect_sca_1.py`, `detect_sca_2.py`: Used for scalability experiment

Furthermore, tables\ and figures\ contain the results of rq1.

(1) tables\\: this directory contains the tables that present the detailed experimental results, including recall, precision, and F1 scores for each combination of code metrics tested in our experiment.

(2)figures\\: this directory contains the performance analysis charts that visually represent the results for each combination of code metrics. 

This is the README.md file for the code of the paper:

xxxxx

Abstract: Code clone detection is about finding out similar code fragments, which has drawn much attention in software engineering since it is important for software maintenance and evolution. Researchers have proposed many techniques and tools for source code clone detection, but current detection methods concentrate on analyzing or processing code samples individually without exploring the underlying connections among code samples. In this paper, we propose Gitor to capture the underlying connections among different code samples. Specifically, given a source code database, we first tokenize all code samples to extract the pre-defined individual information (i.e., keywords and code metrics). After obtaining all samples’ individual information, we leverage them to build a large global sample graph where each node is a code sample or a type of individual information. Then we apply a node embedding technique on the global sample graph to extract all the samples’ vector representations. After collecting all code samples’ vectors, we can either directly compute the cosine similarity between any two samples to quickly detect potential clone pairs or input these vectors into a neural network classifier to identify code clones. To demonstrate the effectiveness of Gitor, we evaluate it on a widely used dataset namely BigCloneBench. Our experimental results show that Gitor has higher accuracy in terms of code clone detection and excellent execution time for input of one million code pairs compared to existing state-of-the-art tools. Moreover, we also evaluate multiple code metrics through grouped experiments, systematically assessing the detection performance of various metric combinations. The results show that code metrics can effectively detect code clones.

We welcome citations of our paper. If you find Gitor useful for your research, please consider citing it using the following Bibtex entry:
xxxx
