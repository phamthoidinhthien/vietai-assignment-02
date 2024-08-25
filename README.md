# Data Tracking Answer Agent

This agent can answer info of event tracking or search for page that contain the event tracks. It purpose to help people easily search for event track via natural chat with a bot. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)


## Installation

1. **Clone the repo**
2. **Create a virtual environment and activate it:**
    ```sh
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Upgrade `pip` and install the required packages:**
    ```sh
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
4. **Input OPENAI_API_KEY in .env file**

## Usage

1. **Run the main script:**

    ```sh
    python main.py
    ```

2. **Interact with the agent via the Gradio interface.**

## Configuration
  - The model and embedding settings are configured in the `Settings` object:
    ```python
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    ```