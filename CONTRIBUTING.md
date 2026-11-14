# Contributing to Gemma RAG AI Server

Thank you for your interest in contributing to the Gemma RAG AI Server! We welcome bug reports, feature suggestions, documentation enhancements, and pull requests.

## Development Setup

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/gemma-rag.git
   cd gemma-rag
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Prepare Model Binaries**

   Place compatible `.litertlm` models into the `models/` directory:
   - `models/gemma-4-E2B-it.litertlm`
   - `models/embeddinggemma-2-740m.litertlm`

## Branching & Commit Guidelines

- **Branch Naming**: Use descriptive branch names:
  - `feat/feature-name` for new features
  - `fix/bug-description` for bug fixes
  - `docs/doc-update` for documentation changes
  - `refactor/component-name` for code restructuring

- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat: add model token count tracking`
  - `fix: handle websocket disconnection cleanly`
  - `docs: update endpoint documentation`
  - `refactor: modularize routing modules`

## Pull Request Process

1. Ensure your code conforms to PEP 8 standards.
2. Verify local execution with `uvicorn main:app --reload`.
3. Push your branch to your fork and submit a Pull Request to `main`.
4. Provide a clear description of the problem solved and any relevant context or reproduction steps.

