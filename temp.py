from groq import Groq
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
import tempfile
from datetime import datetime

# Read Groq API key from file
with open("groq_api_key.txt", "r") as f:
    groq_api_key = f.read().strip()

client = Groq(api_key=groq_api_key)

def generate_response(prompt, temperature, model="llama3-70b-8192"):
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

def generate_responses_parallel(prompt, temperatures, model="llama3-70b-8192"):
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

def score_response(prompt, response_text, temperature, model="llama3-70b-8192"):
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
            temperature=0,
            max_tokens=10,
            top_p=1,
            stream=False,
        )

        score_text = completion.choices[0].message.content.strip()
        match = re.search(r"\b(\d+)\b", score_text)
        if match:
            score = int(match.group(1))
            return min(100, max(0, score))
        return 50
    except:
        return 50

def rank_responses(prompt, responses, model="llama3-70b-8192"):
    """Score and rank all responses."""
    valid_responses = [r for r in responses if r["status"] == "success"]

    if not valid_responses:
        return responses

    print("Scoring responses...")

    for response in valid_responses:
        response["score"] = score_response(
            prompt, response["response"], response["temperature"], model
        )

    for response in responses:
        if response["status"] == "error":
            response["score"] = 0

    ranked_responses = sorted(responses, key=lambda x: x["score"], reverse=True)

    for i, response in enumerate(ranked_responses):
        response["rank"] = i + 1

    return ranked_responses

def generate_html_report(prompts_data, overall_stats):
    """Generate an HTML report."""

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoTemp Multi-Prompt Analysis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: sans-serif; background: #f4f7f6; padding: 20px; }
            .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: #4CAF50; color: white; padding: 20px; text-align: center; }
            .header h1 { font-size: 1.8rem; margin-bottom: 5px; }
            .overall-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; padding: 20px; background: #e8f5e9; border-bottom: 1px solid #dcdcdc; }
            .stat-card { background: white; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .stat-number { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
            .stat-label { color: #555; font-size: 0.8rem; margin-top: 5px; }
            .prompt-section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .prompt-section:last-child { border-bottom: none; }
            .prompt-header { background: #f0f0f0; padding: 15px; margin: 0 20px 15px 20px; border-radius: 5px; }
            .prompt-title { font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; }
            .prompt-text { font-size: 0.95rem; line-height: 1.5; }
            .prompt-stats { display: flex; justify-content: space-around; margin: 0 20px 15px 20px; padding: 15px; background: #f8f8f8; border-radius: 5px; }
            .prompt-stat { text-align: center; }
            .prompt-stat-number { font-size: 1.2rem; font-weight: bold; color: #4CAF50; }
            .prompt-stat-label { color: #777; font-size: 0.75rem; margin-top: 5px; }
            .responses-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; padding: 0 20px; }
            .response-card { background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #ddd; }
            .response-card.rank-1 { border-color: #ffd700; }
            .response-card.rank-2 { border-color: #c0c0c0; }
            .response-card.rank-3 { border-color: #cd7f32; }
            .response-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: #f0f0f0; border-bottom: 1px solid #ddd; }
            .rank-badge { background: #4CAF50; color: white; padding: 4px 8px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
            .rank-1 .rank-badge { background: #ffd700; color: #333; }
            .rank-2 .rank-badge { background: #c0c0c0; color: #333; }
            .rank-3 .rank-badge { background: #cd7f32; color: white; }
            .temp-score { display: flex; gap: 8px; align-items: center; font-size: 0.8rem; }
            .temperature { background: #e0e0e0; padding: 2px 6px; border-radius: 4px; font-weight: 500; color: #444; }
            .score { font-weight: bold; color: #4CAF50; }
            .response-content { padding: 15px; line-height: 1.5; color: #333; font-size: 0.9rem; max-height: 200px; overflow-y: auto; }
            .error-response { background: #ffebee; color: #c62828; border-color: #ef9a9a; }
            .timestamp { text-align: center; color: #777; font-size: 0.8rem; padding: 15px; background: #f0f0f0; }
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
                    <span class="stat-number">{{TOTAL_PROMPTS}}</span>
                    <div class="stat-label">Total Prompts</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{{TOTAL_RESPONSES}}</span>
                    <div class="stat-label">Total Responses</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{{TEMPS_TESTED}}</span>
                    <div class="stat-label">Temperatures Tested</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{{AVG_BEST_TEMP}}</span>
                    <div class="stat-label">Avg Best Temperature</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{{SUCCESS_RATE}}%</span>
                    <div class="stat-label">Overall Success Rate</div>
                </div>
            </div>

            {{PROMPTS_CONTENT}}

            <div class="timestamp">
                Generated on {{TIMESTAMP}} | Temperatures tested: {{TEMPERATURE_RANGE}}
            </div>
        </div>
    </body>
    </html>
    """

    prompts_content = ""
    for i, prompt_data in enumerate(prompts_data):
        prompt = prompt_data["prompt"]
        responses = prompt_data["responses"]

        response_cards = ""
        for response in responses:
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

            card_html = f"""
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
            response_cards += card_html

        successful = [r for r in responses if r["status"] == "success"]
        best_temp = responses[0]["temperature"] if responses else 0
        best_score = responses[0]["score"] if responses else 0
        avg_score = sum(r["score"] for r in successful) / len(successful) if successful else 0

        prompt_section = f"""
        <div class="prompt-section">
            <div class="prompt-header">
                <div class="prompt-title">
                    Prompt #{i+1}
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
        prompts_content += prompt_section

    html = html_template.replace("{{TOTAL_PROMPTS}}", str(overall_stats["total_prompts"]))
    html = html.replace("{{TOTAL_RESPONSES}}", str(overall_stats["total_responses"]))
    html = html.replace("{{TEMPS_TESTED}}", str(overall_stats["temps_tested"]))
    html = html.replace("{{AVG_BEST_TEMP}}", str(overall_stats["avg_best_temp"]))
    html = html.replace("{{SUCCESS_RATE}}", str(overall_stats["success_rate"]))
    html = html.replace("{{PROMPTS_CONTENT}}", prompts_content)
    html = html.replace("{{TIMESTAMP}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{TEMPERATURE_RANGE}}", str(overall_stats["temperature_range"]))

    return html

def autotemp_multi_prompt(prompts, temperatures, model="llama3-70b-8192"):
    """
    Generate responses for multiple prompts and create a comprehensive HTML report.
    """
    print(f"Testing {len(prompts)} prompts with {len(temperatures)} temperatures each...")
    print(f"Temperatures: {temperatures}")

    all_prompts_data = []
    total_responses = 0
    total_successful = 0
    all_best_temps = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Processing Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:70]}...") # Truncate for display

        responses = generate_responses_parallel(prompt, temperatures, model=model)
        ranked_responses = rank_responses(prompt, responses, model=model)

        all_prompts_data.append({"prompt": prompt, "responses": ranked_responses})

        total_responses += len(responses)
        total_successful += len([r for r in responses if r["status"] == "success"])

        if ranked_responses:
            all_best_temps.append(ranked_responses[0]["temperature"])

    overall_stats = {
        "total_prompts": len(prompts),
        "total_responses": total_responses,
        "temps_tested": len(temperatures),
        "avg_best_temp": round(sum(all_best_temps) / len(all_best_temps), 1)
        if all_best_temps
        else 0,
        "success_rate": int((total_successful / total_responses) * 100)
        if total_responses > 0
        else 0,
        "temperature_range": f"{min(temperatures)}-{max(temperatures)}",
    }

    print("\nGenerating HTML report...")
    html_content = generate_html_report(all_prompts_data, overall_stats)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        html_file = f.name

    print(f"HTML report saved to: {html_file}")
    webbrowser.open(f"file://{html_file}")

    return all_prompts_data, html_file

if __name__ == "__main__":
    prompts = [
        "Write a creative short story about a robot learning to cook.",
        # "Explain quantum computing to a 5-year-old.",
        # "Create a business plan for a sustainable coffee shop.",
        # "Write a poem about the ocean in the style of Shakespeare.",
        "Describe the future of artificial intelligence in 2030.",
    ]
    temperatures = [0.1, 0.3, 0.5]

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
        best = result["responses"][0]
        print(f"Prompt {i+1}: Best temp = {best['temperature']}, Score = {best['score']}/100")
