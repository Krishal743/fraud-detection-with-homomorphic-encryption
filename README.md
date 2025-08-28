Project Description

This project is a Privacy-Preserving Federated Learning System for Fraud Detection.

In the real world, multiple banks face the challenge of detecting fraudulent transactions, but sharing raw transaction data is not possible due to privacy laws, security concerns, and competition.

To solve this, we use Federated Learning (FL):

Each bank trains its own local fraud detection model on its private data.

Instead of sending data, banks send only their model parameters (weights/gradients) to a central aggregator.

The central server combines (averages) these updates to build a global fraud detection model, which is then redistributed back to the banks.

This way:

Banks benefit from collective intelligence without exposing sensitive data.

Fraud detection models become more robust because they learn from diverse patterns across institutions.

The design supports scaling to multiple banks and different datasets.

For demonstration, the project uses publicly available datasets like Credit Card Fraud (European dataset) and PaySim (mobile money simulator). The pipeline is modular, so datasets are preprocessed, sharded into "banks," and then fed into both baseline local training and federated training for comparison.

Privacy-Preserving Extension (Planned):
The project also lays the groundwork for integrating Homomorphic Encryption (HE) to protect model updates during transmission, preventing potential leakage from gradients or weights.

In one line:
This project demonstrates how federated learning can enable multiple banks to collaboratively train a fraud detection model without ever sharing raw transaction data.
