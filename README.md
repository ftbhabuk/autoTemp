# AutoTemp: Multi-Prompt AI Response Analyzer

AutoTemp is a Python tool that generates AI responses for multiple prompts across a range of temperatures using the Groq API, scores them for quality, and produces a detailed HTML report ranking the results.

## Features
- Tests multiple prompts with various temperatures in parallel.
- Scores responses based on relevance, clarity, usefulness, and creativity.
- Generates an HTML report with ranked responses, stats, and visualizations.
- Displays the model used (e.g., `llama3-70b-8192`) dynamically in the report.
- Opens the report in your default browser (optional).

## Prerequisites
- **Python 3.8+**
- **Groq API Key**: Obtain from [groq](https://console.groq.com/keys).
- **Dependencies**:
  - `groq`: For interacting with the Groq API.
  - Install via pip:
    ```bash
    pip install groq
    ```

## Setup
1. **Clone or Download**:
   - Download the `autotemp.py` script or clone the repository.

2. **Set Up Groq API Key**:
   - **Option 1**: Add to `.env` file in the project directory:
     ```plaintext
     GROQ_API_KEY=your_api_key_here
     ```
   - **Option 2**: Set as environment variable:
     ```bash
     export GROQ_API_KEY=your_api_key_here
     ```
   

3. **Install Dependencies**:
   ```bash
   pip install groq
   ```

## Usage
Run the script with Python, specifying prompts, temperatures, and optionally the model or browser behavior.

### Basic Command
```bash
python autotemp.py
```
- Runs with default prompts and temperatures (`[0.1, 0.3, 0.5, 0.7, 0.9]`).
- Uses `llama3-70b-8192` model by default.
- Generates an HTML report and opens it in your browser.

### Custom Prompts and Temperatures
Modify the `prompts` and `temperatures` lists in `autotemp.py`:
```python
prompts = [
    "Write a short story about a robot learning to cook.",
    "Create a business plan for a coffee shop during a crisis."
]
temperatures = [0.1, 0.5, 1.0]
```

Then run:
```bash
python autotemp.py
```

### Specify a Different Model
Edit the model in the script or pass it programmatically (if extended). For example:
```python
results, html_file = autotemp_multi_prompt(prompts, temperatures, model="mixtral-8x7b-32768")
```

### Disable Browser Auto-Open
Set `open_browser=False`:
```python
results, html_file = autotemp_multi_prompt(prompts, temperatures, open_browser=False)
```

### Example Output
- The script generates responses, scores them (0-100), and ranks them.
- An HTML report (`*.html` in a temp directory) is created, showing:
  - Model used (e.g., `llama3-70b-8192`).
  - Total prompts, responses, temperatures tested, average best temperature, and success rate.
  - Per-prompt stats: best temperature, best score, average score, success rate.
  - Ranked response cards with temperature, score, and content.

## Example
```bash
python autotemp.py
```
- Tests 2 default prompts with 5 temperatures (10 responses total).
- Outputs: `✅ HTML report saved to: /tmp/tmp12345.html`
- Opens the report in your browser (if `open_browser=True`).

## Notes
- Ensure a valid Groq API key to avoid errors.
- The script uses parallel processing (`ThreadPoolExecutor`) for efficiency.
- Scoring uses the same model with `temperature=0` for consistency.
- To debug temperature bias (e.g., 0.1 often best), inspect raw responses or adjust scoring criteria in `score_response`.


