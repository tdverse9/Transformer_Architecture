# Transformer

A from-scratch implementation of the Transformer architecture in Python, built for learning and experimentation with the core building blocks introduced in *"Attention Is All You Need"* (Vaswani et al., 2017).

## Overview

This project implements the key components of the Transformer model, including token/positional embeddings and the encoder stack, with a test script to validate the model's forward pass.

## Project Structure

```
Transformer/
├── main/
│   ├── __init__.py
│   ├── embedding.py      # Token & positional embedding layers
│   ├── encoder.py        # Transformer encoder block(s)
│   └── test_model.py     # Script to test/validate the model
├── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- **Embedding Layer** (`embedding.py`) — Converts input tokens into dense vector representations and injects positional information.
- **Encoder** (`encoder.py`) — Implements the Transformer encoder, including multi-head self-attention and feed-forward sublayers.
- **Model Testing** (`test_model.py`) — Runs sample input through the model to verify shapes and outputs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Transformer.git
   cd Transformer
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the test script to see the model in action:

```bash
python main/test_model.py
```

## Requirements

See [`requirements.txt`](./requirements.txt) for the full list of dependencies.

## Roadmap

- [ ] Add decoder implementation
- [ ] Add training loop and example dataset
- [ ] Add attention visualization
- [ ] Add unit tests

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (add one if not already present).
