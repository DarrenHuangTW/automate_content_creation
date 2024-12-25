import aisuite as ai
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import quote
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
config = {
    "semrush_database": "us",
    "semrush_display_limit": 50,
    "semrush_display_filter": "%2B%7CPo%7CLt%7C50",
    "semrush_display_sort": "po_asc",
    "jina_api_timeout": 15,
    "openai_model": "openai:gpt-4o-mini",
    "openai_temperature": 0.8
}

# --- Initialization ---
client = ai.Client()
load_dotenv()

# --- Helper Functions ---

def handle_api_errors(response, api_name):
    if response.status_code != 200:
        logging.error(f"Failed to retrieve data from {api_name}. Status code: {response.status_code}")
        return False
    return True

# --- Step 2: SerpAPI Data Retrieval ---

def get_serpapi_data(topic_query):
    base_url = "https://serpapi.com/search.json"
    params = {
        "q": topic_query,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": os.getenv("SERPAPI_KEY")
    }
    response = requests.get(base_url, params=params)
    if handle_api_errors(response, "SerpAPI"):
        results = response.json()
        data = []
        for result in results.get('organic_results', []):
            data.append({
                'Position': result.get('position'),
                'Link': result.get('link'),
                'Title': result.get('title')
            })
        df_serp = pd.DataFrame(data)
        logging.info("Successfully retrieved SerpAPI data.")
        return df_serp
    else:
        return pd.DataFrame()

# --- Step 3: SEMRush Data Retrieval and Processing ---

def get_semrush_data(url, api_key=os.getenv("SEMRUSH_API_KEY")):
    base_url = "https://api.semrush.com/"
    type_param = "url_organic"
    export_columns = "Ph,Po,Nq,Cp,Co"
    full_url = (
        f"{base_url}?type={type_param}&key={api_key}"
        f"&display_limit={config['semrush_display_limit']}&export_columns={export_columns}"
        f"&url={quote(url)}&database={config['semrush_database']}"
        f"&display_filter={config['semrush_display_filter']}&display_sort={config['semrush_display_sort']}"
    )
    response = requests.get(full_url)
    if handle_api_errors(response, "SEMRush"):
        decoded_output = response.content.decode('utf-8')
        lines = decoded_output.split('\r\n')
        headers = lines[0].split(';')
        json_data = []
        for line in lines[1:]:
            if line:
                values = line.split(';')
                record = {header: value for header, value in zip(headers, values)}
                json_data.append(record)
        return json_data
    else:
        return []

def process_semrush_data(df_results):
    df_results['SEMRush_Data'] = df_results['Link'].apply(get_semrush_data)
    logging.info("Successfully retrieved SEMRush data.")
    all_keywords = []
    for data in df_results['SEMRush_Data']:
        if data:
            all_keywords.extend([item['Keyword'] for item in data])

    keyword_counts = Counter(all_keywords)
    highest_count = max(keyword_counts.values())
    second_highest_count = sorted(set(keyword_counts.values()), reverse=True)[1] if len(set(keyword_counts.values())) > 1 else 0
    top_keywords = [keyword for keyword, count in keyword_counts.items() if count == highest_count or count == second_highest_count]

    if highest_count == 2:
        top_keywords = [keyword for keyword, count in keyword_counts.items() if count in [1, 2]]

    search_volume_keywords = sorted(
        [(item['Keyword'], int(item['Search Volume'])) for data in df_results['SEMRush_Data'] if data for item in data],
        key=lambda x: x[1],
        reverse=True)[:10]

    final_keywords = set(top_keywords + [keyword for keyword, _ in search_volume_keywords])
    final_keywords_df = pd.DataFrame(
        [(keyword, next((item['Search Volume'] for data in df_results['SEMRush_Data'] if data for item in data if item['Keyword'] == keyword), 0))
         for keyword in final_keywords],
        columns=['Keyword', 'Search Volume'])
    logging.info("Successfully processed SEMRush data and extracted keywords.")
    return final_keywords_df

# --- Step 4: Content Fetching ---

def fetch_content(url):
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Retain-Images': 'none',
        "Accept": "application/json",
        'X-Timeout': str(config["jina_api_timeout"])
    }
    try:
        response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
        if handle_api_errors(response, "Jina AI Reader"):
            response_json = response.json()
            if response_json['code'] == 200:
                logging.info(f"Successfully fetched content from {url}.")
                return response_json['data']['content']
            else:
                logging.warning(f"Jina API error for {url}: {response_json.get('error', 'Unknown error')}")
                return f"ERROR: {url} blocks Jina API or other error occurred."
        else:
            return f"ERROR: Failed to use Jina API for {url}."

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching content from {url}: {e}")
        return f"ERROR: Request failed for {url}."
    except Exception as e:
        logging.error(f"Unknown error fetching content from {url}: {e}")
        return f"ERROR: Unknown error processing {url}."

# --- Step 5-10: AI Model Interactions ---

def interact_with_ai(messages, model=config["openai_model"], temperature=config["openai_temperature"]):
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        result = response.choices[0].message.content
        return result
    except Exception as e:
        logging.error(f"Error during AI interaction: {e}")
        return None

# --- Main Workflow ---

def main():
    # Step 1: Select a Topic
    topic_query = input("Enter the topic you want to write about: ") # Direct topic input
    
    # Step 2: Retrieve SERP Data
    df_serp = get_serpapi_data(topic_query)


    # Step 3: Retrieve SEMrush Data
    # Step 4: Retrieve Onpage Content
    if not df_serp.empty:
        df_results = df_serp.copy()
        final_keywords_df = process_semrush_data(df_results)
        df_results['Content'] = df_results['Link'].apply(fetch_content)

        # Step 5: Content Analysis
        messages_content_analysis = [
            {"role": "system", "content": "You are a meticulous content researcher with expertise in analyzing web content, particularly articles and blogs. You have access to a list of webpage contents related to the topic a user is interested in."},
            {"role": "user", "content": f"Analyze the provided content below. First, determine if each piece of content is a blog or an article. Disregard any content that is not a blog or an article. For each identified blog or article, add it to a review list. Then, thoroughly review each item on this list and provide an analysis that includes: (1) Common topics and subtopics covered across these blogs/articles. (2) Any contradicting viewpoints among the top 10 results. (3) For users searching for '{topic_query}', identify information gaps - what are they likely interested in that isn't covered, or what questions might they have that remain unanswered by these sources?\n\n"
                                    + "\n".join([f"WEB CONTENT {i + 1}\n{content}" for i, content in enumerate(df_results['Content']) if content])
            }
        ]

        content_analysis = interact_with_ai(messages_content_analysis)
        print(f"Content Analysis:\n{content_analysis}\n")

        # Step 6: Generate Content Plan
        messages_content_plan = [
            {"role": "system", "content": "You are an expert content strategist skilled in crafting detailed and actionable content plans. You are adept at creating outlines that are clear, comprehensive, and tailored to the specific needs of a given topic. You have access to a detailed analysis of competitor content related to the topic a user is interested in."},
            {"role": "user", "content": f"Considering the content analysis provided, develop a comprehensive content plan. The plan should include:\n\n"
                                    f"Topic: {topic_query}\n"
                                    f"An outline with a hierarchical structure of headings and subheadings that logically organize the content.\n\n"
                                    f"Incorporate these SEO keywords: {final_keywords_df}. Ensure these keywords are naturally integrated into the headings and subheadings where relevant.\n\n"
                                    f"While developing the plan, make sure to:\n"
                                    f"Address the common topics and subtopics identified in the content analysis.\n"
                                    f"Highlight any areas with contradicting viewpoints, and suggest a balanced approach to these topics.\n\n"
                                    f"CONTENT ANALYSIS:\n {content_analysis}"
            }
        ]
        content_plan = interact_with_ai(messages_content_plan)
        print(f"Content Plan:\n{content_plan}\n")

        # Step 7: Generate Content Draft
        messages_content_draft = [
            {"role": "system", "content": "You are a skilled content writer specializing in crafting engaging, informative, and SEO-friendly blog posts. You excel at following detailed content plans and adapting your writing style to meet specific guidelines and objectives. You have access to a content plan and an analysis of competitor content related to the topic a user is interested in."},
            {"role": "user", "content": f"Using the provided Content Plan and the insights from the Competitor Content Analysis, write a comprehensive article. Focus on delivering high-quality content that is engaging, informative, and optimized for search engines. Adhere to the structure and guidelines set in the Content Plan, and ensure the article addresses the topics and keywords specified. The article should be written in a style that is accessible and appealing to the target audience, while also being mindful of SEO best practices. Please provide the article only, without any additional commentary or explanations.\n\n"
                                    f"Content Plan:\n {content_plan}\n\n"
                                    f"Competitor Content Analysis:\n {content_analysis}"
            }
        ]
        content_draft = interact_with_ai(messages_content_draft)
        print(f"Content Draft:\n{content_draft}\n")

        # Step 8: Proofread the Draft Post
        messages_proofread_draft = [
            {"role": "system", "content": "You are an expert content editor with a keen eye for detail, specializing in refining and polishing written content. You excel at ensuring content is engaging, error-free, and adheres to SEO best practices. You have access to a draft article, its corresponding content plan, and an analysis of competitor content related to the topic a user is interested in."},
            {"role": "user", "content": f"Review the provided Content Draft, ensuring it aligns with the Content Plan and surpasses the quality of competitor content as detailed in the Competitor Content Analysis. Your task is to refine the draft, focusing on enhancing its engagement, clarity, and readability. Ensure the content is free of grammatical errors, follows SEO best practices, and is well-structured. Make any necessary adjustments to improve the overall quality and impact of the article. Please provide the revised article only, without any additional commentary or explanations.\n\n"
                                    f"Content Draft:\n {content_draft}\n\n"
                                    f"Content Plan:\n {content_plan}\n\n"
                                    f"Competitor Content Analysis:\n {content_analysis}"
            }
        ]
        proofread_draft = interact_with_ai(messages_proofread_draft)
        print(f"Proofread Draft:\n{proofread_draft}\n")

        # Step 9: SEO Recommendations
        messages_seo_recommendations = [
            {"role": "system", "content": "You are a seasoned SEO expert specializing in optimizing blog articles for search engines. You are adept at crafting compelling title tags and meta descriptions that improve click-through rates and accurately reflect the content. You have access to the final version of a blog article and a list of its targeting keywords related to the topic a user is interested in."},
            {"role": "user", "content": f"Examine the provided Content and the list of Targeting Keywords. Develop an optimized URL slug for the article. Generate three variations of a Title Tag, each designed to capture attention and encourage clicks. Additionally, create three variations of a Meta Description that accurately summarize the article's content and entice users to read further. Ensure each suggestion is SEO-friendly and aligns with current best practices. Please provide only the URL slug, Title Tags, and Meta Descriptions, without any additional commentary or explanations.\n\n"
                                    f"Content:\n {proofread_draft}\n\n"
                                    f"Targeting Keywords:\n {final_keywords_df}\n"
            }
        ]
        seo_recommendations = interact_with_ai(messages_seo_recommendations)
        print(f"SEO Recommendations:\n{seo_recommendations}\n")

        # Step 10: Final Deliverable
        messages_final_deliverable = [
            {"role": "system", "content": "You are a meticulous Senior Project Manager with expertise in presenting comprehensive project deliverables. You excel at organizing and summarizing complex information into a clear, concise, and client-ready format. You have access to all the outputs generated during a content creation process related to the topic a user is interested in."},
            {"role": "user", "content": f"Compile the following information into a well-structured document for client presentation. The document should clearly outline the entire content generation process and include: \n\n"
                                    f"- Title & Meta Description: Present the SEO-optimized title and meta description options, highlighting the chosen or recommended ones. Include alternative options for consideration.\n"
                                    f"- URL: Provide the finalized URL slug for the article.\n"
                                    f"- Targeting Keywords: List the primary keywords targeted in the content, along with their search volume.\n"
                                    f"- Competitors: Summarize key information about the top competitors (Position, Link, and Title only), derived from the SERP analysis.\n"
                                    f"- Notes: Offer insights into the content strategy, explaining what aspects are covered, unique points not addressed by competitors, and areas that may require human validation or review for accuracy and completeness.\n"
                                    f"- Final Content: Present the fully proofread and polished article.\n\n"
                                    f"Ensure the deliverable is client-friendly, easy to understand, and provides a comprehensive overview of the project. Please provide the final deliverable document only, without any additional commentary or explanations.\n\n"
                                    f"Content:\n {proofread_draft}\n\n"
                                    f"SEO Recommendations:\n {seo_recommendations}\n"
                                    f"Targeting Keywords:\n {final_keywords_df}\n"
                                    f"Competitors:\n {df_serp}\n"
                                    f"Competitors Analysis:\n {content_analysis}\n"
            }
        ]
        final_deliverable = interact_with_ai(messages_final_deliverable)
        print(f"Final Deliverable:\n{final_deliverable}\n")
    else:
        print("Could not retrieve essential data. Exiting.")

if __name__ == "__main__":
    main()