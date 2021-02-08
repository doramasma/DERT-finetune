# Fine-tuning Detection Transformer (DERT)

The mainly purpose of this repository is to fine-tune Facebook's [DERT](https://github.com/facebookresearch/detr) (DEtection Transformer). 


![alt text](./images/dert-result.png "Dert result after finetune")

Author: Doramas Báez Bernal <br/>
Email: doramas.baez101@alu.ulpgc.es

## Index

* [Introduction](#Introduction)
* [Requirements](#Requirements) 
* [Detection Transformer (DERT)](#Dert)
    * [General information](#GeneralInformation)
    * [Fine-tuning](#Fine-tuning)
* [References](#References)

## Introduction <a id="Introduction"></a>

Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

## Requirements <a id="Requirements"></a>

## Detection Transformer (DERT) <a id="Dert"></a>

### General information (DERT) <a id="GeneralInformation"></a>

### Fine-tuning <a id="Fine-tuning"></a>

## References <a id="References"></a> 