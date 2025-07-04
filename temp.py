import os
import re
import tempfile
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from groq import Groq

# Initialize Groq client
try:
    # It's safer to load API keys from environment variables
    # as it prevents them from being committed to version control.
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # Fallback to file if environment variable is not set
        with open("groq_api_key.txt", "r") as f:
            groq_api_key = f.read().strip()
except FileNotFoundError:
    raise ValueError(
        "GROQ_API_KEY environment variable not set and groq_api_key.txt not found."
    )
except Exception as e:
    raise ValueError(f"Error loading Groq API key: {e}")

client = Groq(api_key=groq_api_key)


def generate_response(prompt, temperature, model="llama-3.1-8b-instant"):
    """Generate a single response at a given temperature."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
            top_p=1,
            stream=False,
        )
        return {
            "temperature": temperature,
            "response": completion.choices[0].message.content.strip(),
            "status": "success",
        }
    except Exception as e:
        return {
            "temperature": temperature,
            "response": f"Error: {e}",
            "status": "error",
        }


def generate_responses_parallel(prompt, temperatures, model="llama-3.1-8b-instant"):
    """Generate responses in parallel for faster execution."""
    responses = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_temp = {
            executor.submit(generate_response, prompt, temp, model): temp
            for temp in temperatures
        }

        for future in as_completed(future_to_temp):
            result = future.result()
            responses.append(result)

    return sorted(responses, key=lambda x: x["temperature"])


def score_response(prompt, response_text, temperature, model="llama-3.1-8b-instant"):
    """Score a single response out of 100."""
    score_prompt = f"""
    Rate this response to the prompt on a scale of 0-100 considering:
    - Relevance to the prompt
    - Clarity and readability
    - Usefulness and completeness
    - Creativity (if appropriate)

    Prompt: "{prompt}"
    Response: "{response_text}"
    Temperature used: {temperature}

    Reply with ONLY the numerical score (0-100).
    """

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": score_prompt}],
            temperature=0,  # Use 0 temperature for consistent scoring
            max_tokens=10,
            top_p=1,
            stream=False,
        )

        score_text = completion.choices[0].message.content.strip()
        match = re.search(r"\b(\d+)\b", score_text)
        if match:
            score = int(match.group(1))
            return min(100, max(0, score))
        # If no clear score is parsed, return 0 as it indicates an issue
        return 0
    except Exception as e:
        # Return 0 if there's an error during scoring
        print(f"Error scoring response: {e}")
        return 0


def rank_responses(prompt, responses, model="llama-3.1-8b-instant"):
    """Score and rank all responses."""
    # Separate successful and error responses
    successful_responses = [r for r in responses if r["status"] == "success"]
    error_responses = [r for r in responses if r["status"] == "error"]

    if not successful_responses:
        # If no successful responses, all responses are errors or empty
        for response in error_responses:
            response["score"] = 0
            response["rank"] = 0  # Indicate unranked
        return responses

    print("Scoring responses...")

    # Score successful responses
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_response = {
            executor.submit(
                score_response, prompt, r["response"], r["temperature"], model
            ): r
            for r in successful_responses
        }

        for future in as_completed(future_to_response):
            original_response = future_to_response[future]
            original_response["score"] = future.result()

    # Assign score 0 to error responses
    for response in error_responses:
        response["score"] = 0

    # Combine and rank all responses (successful and error)
    all_responses = successful_responses + error_responses
    ranked_responses = sorted(all_responses, key=lambda x: x["score"], reverse=True)

    for i, response in enumerate(ranked_responses):
        response["rank"] = i + 1

    return ranked_responses


def generate_html_report(prompts_data, overall_stats):
    """Generate an HTML report using an f-string for clarity."""

    # Using a multi-line f-string for the HTML template makes it more readable
    # than string.replace(). For very complex templates, a templating engine
    # like Jinja2 would be more suitable.
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoTemp Multi-Prompt Analysis</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: sans-serif; background: #f4f7f6; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
            .header {{ background: #4CAF50; color: white; padding: 20px; text-align: center; }}
            .header h1 {{ font-size: 1.8rem; margin-bottom: 5px; }}
            .overall-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; padding: 20px; background: #e8f5e9; border-bottom: 1px solid #dcdcdc; }}
            .stat-card {{ background: white; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .stat-number {{ font-size: 1.5rem; font-weight: bold; color: #4CAF50; }}
            .stat-label {{ color: #555; font-size: 0.8rem; margin-top: 5px; }}
            .prompt-section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .prompt-section:last-child {{ border-bottom: none; }}
            .prompt-header {{ background: #f0f0f0; padding: 15px; margin: 0 20px 15px 20px; border-radius: 5px; }}
            .prompt-title {{ font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; }}
            .prompt-text {{ font-size: 0.95rem; line-height: 1.5; }}
            .prompt-stats {{ display: flex; justify-content: space-around; margin: 0 20px 15px 20px; padding: 15px; background: #f8f8f8; border-radius: 5px; }}
            .prompt-stat {{ text-align: center; }}
            .prompt-stat-number {{ font-size: 1.2rem; font-weight: bold; color: #4CAF50; }}
            .prompt-stat-label {{ color: #777; font-size: 0.75rem; margin-top: 5px; }}
            .responses-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; padding: 0 20px; }}
            .response-card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #ddd; }}
            .response-card.rank-1 {{ border-color: #ffd700; }}
            .response-card.rank-2 {{ border-color: #c0c0c0; }}
            .response-card.rank-3 {{ border-color: #cd7f32; }}
            .response-header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: #f0f0f0; border-bottom: 1px solid #ddd; }}
            .rank-badge {{ background: #4CAF50; color: white; padding: 4px 8px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }}
            .rank-1 .rank-badge {{ background: #ffd700; color: #333; }}
            .rank-2 .rank-badge {{ background: #c0c0c0; color: #333; }}
            .rank-3 .rank-badge {{ background: #cd7f32; color: white; }}
            .temp-score {{ display: flex; gap: 8px; align-items: center; font-size: 0.8rem; }}
            .temperature {{ background: #e0e0e0; padding: 2px 6px; border-radius: 4px; font-weight: 500; color: #444; }}
            .score {{ font-weight: bold; color: #4CAF50; }}
            .response-content {{ padding: 15px; line-height: 1.5; color: #333; font-size: 0.9rem; max-height: 200px; overflow-y: auto; }}
            .error-response {{ background: #ffebee; color: #c62828; border-color: #ef9a9a; }}
            .timestamp {{ text-align: center; color: #777; font-size: 0.8rem; padding: 15px; background: #f0f0f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AutoTemp Multi-Prompt Analysis</h1>
                <p>AI Response Ranking Across Multiple Prompts & Temperatures</p>
            </div>

            <div class="overall-stats">
                <div class="stat-card">
                    <span class="stat-number">{overall_stats["total_prompts"]}</span>
                    <div class="stat-label">Total Prompts</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{overall_stats["total_responses"]}</span>
                    <div class="stat-label">Total Responses</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{overall_stats["temps_tested"]}</span>
                    <div class="stat-label">Temperatures Tested</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{overall_stats["avg_best_temp"]}</span>
                    <div class="stat-label">Avg Best Temperature</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{overall_stats["success_rate"]}%</span>
                    <div class="stat-label">Overall Success Rate</div>
                </div>
            </div>

            {"".join([_generate_prompt_section(i, pd) for i, pd in enumerate(prompts_data)])}

            <div class="timestamp">
                Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Temperatures tested: {overall_stats["temperature_range"]}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


def _generate_prompt_section(index, prompt_data):
    """Helper function to generate HTML for a single prompt section."""
    prompt = prompt_data["prompt"]
    responses = prompt_data["responses"]

    response_cards = "".join([_generate_response_card(r) for r in responses])

    successful = [r for r in responses if r["status"] == "success"]
    best_temp = responses[0]["temperature"] if responses else 0
    best_score = responses[0]["score"] if responses else 0
    avg_score = (
        sum(r["score"] for r in successful) / len(successful) if successful else 0
    )

    return f"""
    <div class="prompt-section">
        <div class="prompt-header">
            <div class="prompt-title">
                Prompt #{index+1}
            </div>
            <div class="prompt-text">
                {prompt}
            </div>
        </div>

        <div class="prompt-stats">
            <div class="prompt-stat">
                <div class="prompt-stat-number">{best_temp}</div>
                <div class="prompt-stat-label">Best Temperature</div>
            </div>
            <div class="prompt-stat">
                <div class="prompt-stat-number">{best_score}</div>
                <div class="prompt-stat-label">Best Score</div>
            </div>
            <div class="prompt-stat">
                <div class="prompt-stat-number">{avg_score:.1f}</div>
                <div class="prompt-stat-label">Average Score</div>
            </div>
            <div class="prompt-stat">
                <div class="prompt-stat-number">{len(successful)}/{len(responses)}</div>
                <div class="prompt-stat-label">Success Rate</div>
            </div>
        </div>

        <div class="responses-grid">
            {response_cards}
        </div>
    </div>
    """


def _generate_response_card(response):
    """Helper function to generate HTML for a single response card."""
    rank_class = f"rank-{response['rank']}" if response["rank"] <= 3 else ""
    error_class = "error-response" if response["status"] == "error" else ""

    emoji = (
        "🏆"
        if response["rank"] == 1
        else "🥈"
        if response["rank"] == 2
        else "🥉"
        if response["rank"] == 3
        else ""
    )

    return f"""
    <div class="response-card {rank_class} {error_class}">
        <div class="response-header">
            <div class="rank-badge">
                #{response['rank']} {emoji}
            </div>
            <div class="temp-score">
                <div class="temperature">T: {response['temperature']}</div>
                <div class="score">{response['score']}/100</div>
            </div>
        </div>
        <div class="response-content">
            {response['response'].replace('\n', '<br>')}
        </div>
    </div>
    """


def autotemp_multi_prompt(
    prompts, temperatures, model="llama-3.1-8b-instant", open_browser=True
):
    """
    Generate responses for multiple prompts and create a comprehensive HTML report.

    Args:
        prompts (list): A list of strings, each being a prompt.
        temperatures (list): A list of floats representing temperatures to test.
        model (str): The name of the Groq model to use.
        open_browser (bool): If True, opens the generated HTML report in a browser.

    Returns:
        tuple: A tuple containing:
            - all_prompts_data (list): Detailed results for each prompt.
            - html_file (str): The path to the generated HTML report.
    """
    print(f"Testing {len(prompts)} prompts with {len(temperatures)} temperatures each...")
    print(f"Temperatures: {temperatures}")

    all_prompts_data = []
    total_responses_generated = 0
    total_successful_responses = 0
    all_best_temps = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Processing Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:70]}...")  # Truncate for display

        responses = generate_responses_parallel(prompt, temperatures, model=model)
        ranked_responses = rank_responses(prompt, responses, model=model)

        all_prompts_data.append({"prompt": prompt, "responses": ranked_responses})

        total_responses_generated += len(responses)
        total_successful_responses += len(
            [r for r in responses if r["status"] == "success"]
        )

        if ranked_responses and ranked_responses[0]["status"] == "success":
            all_best_temps.append(ranked_responses[0]["temperature"])

    overall_stats = {
        "total_prompts": len(prompts),
        "total_responses": total_responses_generated,
        "temps_tested": len(temperatures),
        "avg_best_temp": round(sum(all_best_temps) / len(all_best_temps), 1)
        if all_best_temps
        else "N/A",
        "success_rate": int(
            (total_successful_responses / total_responses_generated) * 100
        )
        if total_responses_generated > 0
        else 0,
        "temperature_range": f"{min(temperatures)}-{max(temperatures)}",
    }

    print("\nGenerating HTML report...")
    html_content = generate_html_report(all_prompts_data, overall_stats)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        html_file = f.name

    print(f"HTML report saved to: {html_file}")
    if open_browser:
        webbrowser.open(f"file://{html_file}")

    return all_prompts_data, html_file


if __name__ == "__main__":
    prompts = [
        "Write a creative short story about a robot learning to cook.",
        # "Explain quantum computing to a 5-year-old.",
        "layout detailed business plan for surviving a coffee shop during ww3.",
        # "Write a poem about the ocean in the style of Shakespeare.",
        # "Describe the future of artificial intelligence in 2030.",
    ]
    temperatures = [0.1,0.5, 0.7, 0.9]

    print("\n🚀 Starting analysis...")
    print(f"📝 Prompts: {len(prompts)}")
    print(f"🌡️  Temperatures: {temperatures}")
    print(f"🔥 Total responses to generate: {len(prompts) * len(temperatures)}")

    results, html_file = autotemp_multi_prompt(prompts, temperatures)

    print("\n🎉 Analysis complete!")
    print(f"📊 Results saved to: {html_file}")
    print(f"🌐 Report opened in your browser")

    print("\n📈 Quick Summary:")
    for i, result in enumerate(results):
        # Ensure there are responses before trying to access the first one
        if result["responses"]:
            best = result["responses"][0]
            print(
                f"Prompt {i+1}: Best temp = {best['temperature']},"
                f" Score = {best['score']}/100"
            )
        else:
            print(f"Prompt {i+1}: No responses generated.")