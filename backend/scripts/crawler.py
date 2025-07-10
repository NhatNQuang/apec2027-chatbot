import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

# Defines the directory for storing raw crawled data.
RAW_DATA_DIR = "backend/data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Centralized list of URLs to crawl, categorized for better organization.
urls = {
    "About APEC 2025 KOREA": [
        {"url": "https://apec2025.kr/?menuno=89", "page": "APEC"},
        {"url": "https://apec2025.kr/?menuno=90", "page": "Introduction"},
        {"url": "https://apec2025.kr/?menuno=92", "page": "Emblem and Theme"},
        {"url": "https://apec2025.kr/?menuno=93", "page": "Meetings"},
        {"url": "https://apec2025.kr/?menuno=94", "page": "Side Events"}
    ],
    "Visit Korea": [
        {"url": "https://apec2025.kr/?menuno=20", "page": "Korea in Brief"},
        {"url": "https://apec2025.kr/?menuno=22", "page": "Practical Information"},
        {"url": "https://apec2025.kr/?menuno=107", "page": "About Gyeongju"},
        {"url": "https://apec2025.kr/?menuno=137", "page": "Gyeongju Transportation"},
        {"url": "https://apec2025.kr/?menuno=108", "page": "Gyeongju Heritage"},
        {"url": "https://apec2025.kr/?menuno=138", "page": "Gyeongju Attractions"},
        {"url": "https://apec2025.kr/?menuno=141", "page": "Jeju Transportation"},
        {"url": "https://apec2025.kr/?menuno=114", "page": "Jeju Nature & Culture"},
        {"url": "https://apec2025.kr/?menuno=115", "page": "Jeju Themed Travel"},
        {"url": "https://apec2025.kr/?menuno=104", "page": "About Incheon"},
        {"url": "https://apec2025.kr/?menuno=106", "page": "About Busan"},
        {"url": "https://apec2025.kr/?menuno=24", "page": "About Seoul"}
    ]
}

# Standard headers to mimic a web browser and avoid blocking.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_page(url):
    """Fetches HTML content from a given URL and parses it with BeautifulSoup."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

def handle_text_page(contents):
    """Extracts text from pages with a 'text_box03' class structure."""
    text_box = contents.find('div', class_='text_box03')
    if text_box:
        p_tag = text_box.find('p')
        return p_tag.get_text(strip=True) if p_tag else ""
    return ""

def handle_table_page(contents):
    """Extracts event data from pages structured as tables."""
    table = contents.find('table')
    events = []
    if table:
        rows = table.find('tbody').find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                events.append({
                    "no": cols[0].get_text(strip=True),
                    "event": cols[1].get_text(strip=True),
                    "date": cols[2].get_text(strip=True),
                    "venue": cols[3].get_text(strip=True)
                })
    return {"events": events}

def handle_list_page(contents, page_name):
    """
    Processes pages with varying list structures, extracting sub-sections and their content.
    Includes special handling for specific page types to correctly parse data.
    """
    sub_sections = contents.find_all(['h3', 'h4'])
    sub_section_data = []
    processed_h4s = set() # Avoids redundant processing of certain h4 elements

    for sub_section in sub_sections:
        sub_section_title = sub_section.get_text(strip=True)
        
        if sub_section_title == "Useful Information":
            continue
        
        sub_section_content = {"sub_section": sub_section_title, "content": ""}
        
        if page_name in ["Gyeongju Transportation", "Jeju Transportation"]:
            p_tag = sub_section.find_next('p')
            main_content = p_tag.get_text(strip=True) if p_tag else ""
            list_tag = sub_section.find_next(['ol', 'ul'], class_='list_box01')
            items = [li.get_text(strip=True) for li in list_tag.find_all('li')] if list_tag else []
            sub_section_content["content"] = {
                "main_content": main_content,
                "items": items
            }
        
        elif page_name in ["Gyeongju Heritage", "Gyeongju Attractions", "Jeju Nature & Culture", "Jeju Themed Travel"]:
            # Logic to find the correct photo layout section for the current sub_section
            temp_sub_section_data = []
            for layout in contents.find_all('ul', class_='webtong_Photo_layout'):
                # Checks if this layout's h3 matches the current sub_section_title.
                # This is critical for associating content correctly when multiple layouts exist.
                sub_h3 = layout.find('h3')
                if sub_h3 and sub_h3.get_text(strip=True) == sub_section_title:
                    p_tag = sub_h3.find_next('p')
                    main_content = p_tag.get_text(strip=True) if p_tag else ""
                    info_list = sub_h3.find_next('ul', class_='information tourism')
                    info_items = [li.get_text(strip=True) for li in info_list.find_all('li')] if info_list else []
                    temp_sub_section_data.append({
                        "sub_section": sub_section_title,
                        "content": {
                            "main_content": main_content,
                            "information": info_items
                        }
                    })
            if temp_sub_section_data:
                sub_section_data.extend(temp_sub_section_data)
            continue # Prevents redundant processing in the main loop for these types
        
        elif sub_section_title == "Climate & Weather":
            ul_tag = sub_section.find_next('ul', class_='Climate_wrap')
            seasons = []
            if ul_tag:
                for li in ul_tag.find_all('li'):
                    strong = li.find('strong').get_text(strip=True).split('<em>')[0] if li.find('strong') else ""
                    em = li.find('em').get_text(strip=True) if li.find('em') else ""
                    p_text = li.find('p').get_text(strip=True) if li.find('p') else ""
                    seasons.append({
                        "season": strong,
                        "period": em,
                        "description": p_text
                    })
            sub_section_content["content"] = {"seasons": seasons}
        
        elif sub_section_title == "Banking & Currency":
            p_tag = sub_section.find_next('p')
            main_content = p_tag.get_text(strip=True) if p_tag else ""
            sub_section_content["content"] = {"main_content": main_content}
        
        elif sub_section_title == "Electricity and Voltage":
            p_tag = sub_section.find_next('p', class_='mt30')
            main_content = p_tag.get_text(strip=True) if p_tag else ""
            sub_section_content["content"] = main_content
        
        elif sub_section_title in ["Traveler’s Checks", "Credit Cards", "Money Exchange", "Currency Converter"]:
            if sub_section_title not in processed_h4s:
                p_tag = sub_section.find_next('p')
                main_content = p_tag.get_text(strip=True) if p_tag else ""
                sub_section_content["content"] = main_content
        
        elif sub_section_title == "Emergency & Useful Phone Numbers":
            # Gathers all h4s that are children of this h3 section.
            sub_h4s = [h4 for h4 in contents.find_all('h4') if h4.find_previous('h3') == sub_section]
            emergency_data = []
            for h4 in sub_h4s:
                h4_title = h4.get_text(strip=True)
                processed_h4s.add(h4_title)
                ul_tag = h4.find_next('ul', class_='list_box01')
                items = [li.get_text(strip=True) for li in ul_tag.find_all('li')] if ul_tag else []
                emergency_data.append({"sub_section": h4_title, "items": items})
            sub_section_content["content"] = {"sub_sections": emergency_data}
        
        if sub_section_content["content"]:
            sub_section_data.append(sub_section_content)
    
    return sub_section_data

def handle_apec_page(contents):
    """Processes content specific to the 'APEC' overview page."""
    sections = contents.find_all('h3')
    section_data = []
    
    for section in sections:
        section_title = section.get_text(strip=True)
        section_content = {"section": section_title, "content": ""}
        
        if section_title == "What is APEC?":
            about_wrap = section.find_next('div', class_='about_apec_wrap')
            if about_wrap:
                p_tag = about_wrap.find('p')
                section_content["content"] = p_tag.get_text(strip=True) if p_tag else ""
        
        elif section_title in ["Mission", "Vision"]:
            swiper_slide = section.find_parent('div', class_='swiper-slide')
            if swiper_slide:
                p_tag = swiper_slide.find('p')
                section_content["content"] = p_tag.get_text(strip=True) if p_tag else ""
        
        elif section_title == "APEC Member Economies":
            ul_tag = section.find_next('ul', class_='apec_member')
            members = [li.get_text(strip=True) for li in ul_tag.find_all('li')] if ul_tag else []
            note_tag = ul_tag.find_next('p', class_='mt15') if ul_tag else None
            note = note_tag.get_text(strip=True) if note_tag else ""
            section_content["content"] = {"members": members, "note": note}
        
        elif section_title == "APEC in the World":
            p_tag = section.find_next('p')
            section_content["content"] = p_tag.get_text(strip=True) if p_tag else ""
        
        elif section_title == "How APEC operates":
            info_wrap = section.find_next('div', class_='info_wrap')
            if info_wrap:
                p_tag = info_wrap.find('p', class_='mt30')
                section_content["content"] = p_tag.get_text(strip=True) if p_tag else ""
        
        if section_content["content"]:
            section_data.append(section_content)
    
    return section_data

def handle_introduction_and_emblem_page(contents):
    """
    Manages content extraction for 'Introduction' and 'Emblem and Theme' pages,
    handling nested structures and specific content types.
    """
    sections = contents.find_all('h3')
    section_data = []
    
    for section in sections:
        section_title = section.get_text(strip=True)
        section_content = {"section": section_title, "content": ""}
        
        if section_title == "Overview":
            overview_wrap = section.find_next('div', class_='overview_new_inner')
            if overview_wrap:
                ul_tag = overview_wrap.find('ul', class_='overview_text_inner')
                if ul_tag:
                    items = []
                    for li in ul_tag.find_all('li'):
                        strong = li.find('strong').get_text(strip=True) if li.find('strong') else ""
                        em = li.find('em').get_text(strip=True) if li.find('em') else ""
                        items.append({strong: em})
                    section_content["content"] = items
        
        elif section_title == "Emblem of the APEC 2025 KOREA":
            theme_wrap = section.find_next('div', class_='theme_text')
            if theme_wrap:
                section_content["content"] = theme_wrap.get_text(strip=True)
        
        elif section_title == "APEC 2025 KOREA THEME AND PRIORITIES":
            text_box = section.find_next('div', class_='text_box05')
            main_content = text_box.find('p').get_text(strip=True) if text_box and text_box.find('p') else ""
            statement_wrap = section.find_next('ul', class_='statement_wrap')
            statements = []
            if statement_wrap:
                for li in statement_wrap.find_all('li'):
                    strong = li.find('strong').get_text(strip=True) if li.find('strong') else ""
                    p_text = li.find('p').get_text(strip=True) if li.find('p') else ""
                    statements.append({strong: p_text})
            section_content["content"] = {
                "main_content": main_content,
                "statements": statements
            }
        
        elif section_title in ["Korea and APEC", "Korea’s Engagement with APEC", "Korea’s Contribution to APEC"]:
            if section_title == "Korea’s Engagement with APEC":
                li_parent = section.find_parent('li')
                content_text = li_parent.find('p').get_text(strip=True) if li_parent and li_parent.find('p') else ""
                section_content["content"] = content_text
            else:
                p_tag = section.find_next('p')
                content_text = p_tag.get_text(strip=True) if p_tag else ""
                # Adjusting logic to correctly identify photo layouts relevant to this section,
                # avoiding false positives by checking for the parent section.
                photo_layout = section.find_parent('ul', class_='webtong_Photo_layout') or section.find_next('ul', class_='webtong_Photo_layout')
                sub_sections = []
                if photo_layout and section_title != "Korea’s Engagement with APEC":
                    for li in photo_layout.find_all('li'):
                        sub_h3 = li.find('h3')
                        if sub_h3 and sub_h3.get_text(strip=True) != section_title: # Prevents self-inclusion
                            sub_title = sub_h3.get_text(strip=True)
                            sub_p = li.find('p')
                            sub_content = sub_p.get_text(strip=True) if sub_p else ""
                            sub_sections.append({"sub_section": sub_title, "content": sub_content})
                section_content["content"] = {
                    "main_content": content_text,
                    "sub_sections": sub_sections
                }
        
        if section_content["content"]:
            section_data.append(section_content)
    
    return section_data

def crawl_apec_data():
    """
    Main function to orchestrate the crawling process across all defined URLs.
    It dispatches to specific handlers based on the page's structure.
    """
    all_data = []
    
    for category, page_list in urls.items():
        for page_info in page_list:
            url = page_info["url"]
            page_name = page_info["page"]
            
            soup = fetch_page(url)
            if not soup:
                continue
            
            contents = soup.find('div', id='contents')
            if not contents:
                print(f"Error: Could not find div#contents on page {page_name}. Skipping.")
                continue
            
            main_section = contents.find('h2')
            if not main_section:
                print(f"Error: Could not find h2 on page {page_name}. Skipping.")
                continue
            
            main_section_title = main_section.get_text(strip=True)
            section_data = {"section": main_section_title, "content": []}
            
            # Delegates content parsing to specific handlers based on page name.
            if page_name in ["Meetings", "Side Events"]:
                section_data["content"] = handle_table_page(contents)
            elif page_name in ["Korea in Brief", "About Gyeongju", "About Incheon", "About Busan", "About Seoul"]:
                section_data["content"] = handle_text_page(contents)
            elif page_name in ["Gyeongju Transportation", "Jeju Transportation", "Gyeongju Heritage", "Gyeongju Attractions", "Jeju Nature & Culture", "Jeju Themed Travel", "Practical Information"]:
                section_data["content"] = handle_list_page(contents, page_name)
            elif page_name == "APEC":
                section_data["content"] = handle_apec_page(contents)
            elif page_name in ["Introduction", "Emblem and Theme"]:
                section_data["content"] = handle_introduction_and_emblem_page(contents)
            
            if section_data["content"]:
                all_data.append({
                    "category": category,
                    "page": page_name,
                    "sections": [section_data]
                })
    
    return all_data

if __name__ == "__main__":
    crawled_data = crawl_apec_data()
    if crawled_data:
        # Generates a timestamped output filename for raw data.
        output_file = os.path.join(RAW_DATA_DIR, f"apec2025_all_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(crawled_data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {output_file}")
        
        # Prints a summary of the crawled data for immediate verification.
        print("\n--- Crawled Data Summary (First Few Entries) ---")
        for page_entry in crawled_data[:5]: # Displaying first 5 entries for brevity
            print(f"\nCategory: {page_entry['category']}")
            print(f"Page: {page_entry['page']}")
            for section_entry in page_entry['sections']:
                print(f"  Section: {section_entry['section']}")
                # Limit content printout to avoid overwhelming console for large contents
                content_display = str(section_entry['content'])
                if len(content_display) > 200:
                    print(f"  Content: {content_display[:200]}...")
                else:
                    print(f"  Content: {content_display}")
                print("-" * 30) # Shortened separator
            print("=" * 50) # Category separator
    else:
        print("No data was collected from the specified URLs.")