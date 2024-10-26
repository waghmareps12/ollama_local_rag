# Sales Inquiry Assistant

## Overview
The Sales Inquiry Assistant is a chatbot application designed to assist users with inquiries related to sales, rebates, market share, promotions, and payouts. It leverages advanced language models and embeddings to provide accurate and meaningful responses based on the provided context and chat history.

## Features
- **Conversational Memory**: Maintains chat history for context-aware responses.
- **Custom Tools**: Includes tools for checking SAP values and answering general sales inquiries.
- **User-Friendly Interface**: Built with Streamlit for an interactive user experience.

## Requirements
- Python 3.x
- Streamlit
- LangChain
- Ollama
- Pandas
- dotenv

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_TYPE=<your-api-type>
   OPENAI_API_KEY=<your-api-key>
   OPENAI_API_BASE=<your-api-base>
   OPENAI_API_VERSION=<your-api-version>
   ```
2. Run the application:
   ```bash
   streamlit run sales_inquiry_agent-local.py
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
