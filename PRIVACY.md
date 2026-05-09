# Privacy Policy

## Data Handling

This software processes text data for AI-generated content detection and safety classification. When using the LLM ensemble mode, text is sent to third-party API providers.

### Local Mode (LightGBM Features)

- Processes input text entirely locally
- No data transmitted to external services
- All feature extraction runs on the local machine

### LLM Ensemble Mode

When using the multi-LLM ensemble (`--mode llm`), input text (truncated to 500-3000 characters) is sent to the configured API providers:

- **Together.ai** (Llama 3.3, DeepSeek-V3, Qwen3)
- **Groq** (Llama 3.3)
- **Mistral AI** (Mistral Large)
- **Google** (Gemini 2.5 Flash)

Each provider has its own privacy policy. Review their terms before processing sensitive data.

### What This Software Does NOT Do

- Does not collect or store personal information
- Does not track usage or collect analytics
- Does not retain API responses beyond the local cache

### Training Data

Models were trained on the PAN@CLEF 2026 Reasoning Trajectory Detection dataset. No personally identifiable information was used.

## Contact

For privacy concerns, please open an issue on the GitHub repository.
