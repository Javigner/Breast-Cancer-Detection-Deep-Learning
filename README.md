# Breast Cancer Detection: Predict whether the cancer is benign or malignant.

<h3>Cancer Wisconsin (Diagnostic) Data Set</h3>
<h3>Description</h3>

An ANN is based on a collection of connected units or nodes called artificial neurons (analogous to biological neurons in an animal brain). Each connection (synapse) between neurons can transmit a signal from one to another. The receiving (postsynaptic) neuron can process the signal(s) and then signal downstream neurons connected to it. In common ANN implementations, the synapse signal is a real number, and the output of each neuron is calculated by a non-linear function of the sum of its input. Neurons and synapses may also have a weight that varies as learning proceeds, which can increase or decrease the strength of the signal that it sends downstream. Further, they may have a threshold such that only if the aggregate signal is below (or above) that level is the downstream signal sent.

![Image of ANN](https://miro.medium.com/max/2500/1*ZB6H4HuF58VcMOWbdpcRxQ.png)

For more information, [see](https://en.wikipedia.org/wiki/Artificial_neural_network)

Data is given by Fine-needle aspiration (FNA) is a diagnostic procedure used to investigate lumps or masses. In this technique, a thin (23–25 gauge (0.52 to 0.64 mm outer diameter)), hollow needle is inserted into the mass for sampling of cells that, after being stained, are examined under a microscope (biopsy). The sampling and biopsy considered together are called fine-needle aspiration biopsy (FNAB) or fine-needle aspiration cytology (FNAC) (the latter to emphasize that any aspiration biopsy involves cytopathology, not histopathology). Fine-needle aspiration biopsies are very safe minor surgical procedures. Often, a major surgical (excisional or open) biopsy can be avoided by performing a needle aspiration biopsy instead, eliminating the need for hospitalization. In 1981, the first fine-needle aspiration biopsy in the United States was done at Maimonides Medical Center. Today, this procedure is widely used in the diagnosis of cancer and inflammatory conditions.

<h2> How to use it </h2>

```pip3 install -r requirements.txt```
```
usage: Breast_cancer_detector.py [-h] [--Activation_Function ACTIVATION_FUNCTION] Datafile-csv Goal
Breast Cancer Detection

positional arguments:
  Datafile-csv          Your Dataset in csv
  Goal                  training or prediction

optional arguments:
  -h, --help            show this help message and exit
  --Activation_Function ACTIVATION_FUNCTION
                        Choose between sigmoid or softmax
  ```                      
<h3>Test set Accuracy ~ 98%</h3>
