import json
import os

def process_content_node(content_node, metadata_prefix):
    """
    Recursively processes various content types to extract chunks,
    preserving context through metadata. This function ensures semantically
    related content (like main text combined with contact info or related items)
    is grouped into a single chunk.
    """
    chunks = []
    
    if isinstance(content_node, str):
        if content_node.strip():
            chunks.append({
                "content": content_node.strip(),
                "metadata": metadata_prefix.copy()
            })
    
    elif isinstance(content_node, list):
        for item in content_node:
            if isinstance(item, dict):
                # Handles nested sections (e.g., 'What is APEC?', 'Mission')
                if "section" in item and "content" in item:
                    new_metadata_prefix = metadata_prefix.copy()
                    new_metadata_prefix["sub_section"] = item["section"]
                    chunks.extend(process_content_node(item["content"], new_metadata_prefix))
                
                # Handles nested sub_sections (e.g., Korea’s Engagement with APEC)
                elif "sub_section" in item and "content" in item:
                    new_metadata_prefix = metadata_prefix.copy()
                    new_metadata_prefix["sub_section"] = item["sub_section"]
                    chunks.extend(process_content_node(item["content"], new_metadata_prefix))
                
                # Handles specific 'Overview' details when found in a list
                elif "Title" in item and "Location" in item and "Theme and Priorities" in item:
                    overview_content = (
                        f"Title: {item.get('Title', '')}\n"
                        f"Location: {item.get('Location', '')}\n"
                        f"Theme and Priorities: {item.get('Theme and Priorities', '')}"
                    )
                    if overview_content.strip():
                        chunks.append({
                            "content": overview_content.strip(),
                            "metadata": metadata_prefix.copy()
                        })
                else:
                    # Recursively processes other dictionary items within a list
                    chunks.extend(process_content_node(item, metadata_prefix))

    elif isinstance(content_node, dict):
        combined_text_for_this_dict_node = []

        # Prioritizes 'main_content' for combination
        main_content = content_node.get("main_content", "")
        if main_content.strip():
            combined_text_for_this_dict_node.append(main_content.strip())

        # Appends items from 'information' list (e.g., Address, Tel, Website)
        if "information" in content_node and isinstance(content_node["information"], list):
            for info_item in content_node["information"]:
                info_str = str(info_item).strip()
                if info_str:
                    combined_text_for_this_dict_node.append(info_str)

        # Appends items from 'items' list (e.g., City Bus Fare, Taxi Fare)
        if "items" in content_node and isinstance(content_node["items"], list):
            for item_content in content_node["items"]:
                item_str = str(item_content).strip()
                if item_str:
                    combined_text_for_this_dict_node.append(item_str)
        
        # Creates a single chunk from combined text segments. This is a core
        # decision to maintain semantic coherence for certain data types.
        if combined_text_for_this_dict_node:
            chunks.append({
                "content": "\n".join(combined_text_for_this_dict_node),
                "metadata": metadata_prefix.copy()
            })
        
        # Handles nested 'sub_sections'. These typically represent distinct logical units.
        if "sub_sections" in content_node and isinstance(content_node["sub_sections"], list):
            for sub_section_item in content_node["sub_sections"]:
                if "sub_section" in sub_section_item and "content" in sub_section_item:
                    new_metadata_prefix = metadata_prefix.copy()
                    new_metadata_prefix["sub_section"] = sub_section_item["sub_section"]
                    chunks.extend(process_content_node(sub_section_item["content"], new_metadata_prefix))
        
        # Processes specific lists that should generate individual chunks for each item
        # (e.g., events, members, seasons, statements). Each item here is a distinct entity.
        if "seasons" in content_node and isinstance(content_node["seasons"], list):
            for season_data in content_node["seasons"]:
                season_str = f"Season: {season_data.get('season', '')}, Period: {season_data.get('period', '')}, Description: {season_data.get('description', '')}"
                if season_str.strip():
                    new_metadata = metadata_prefix.copy()
                    new_metadata["item_type"] = "season"
                    chunks.append({
                        "content": season_str.strip(),
                        "metadata": new_metadata
                    })
        
        if "members" in content_node and isinstance(content_node["members"], list):
            for member in content_node["members"]:
                if member.strip():
                    new_metadata = metadata_prefix.copy()
                    new_metadata["item_type"] = "member"
                    chunks.append({
                        "content": f"APEC Member Economy: {member.strip()}",
                        "metadata": new_metadata
                    })
            if content_node.get('note', '').strip():
                new_metadata = metadata_prefix.copy()
                new_metadata["item_type"] = "member_note"
                chunks.append({
                    "content": f"Note on APEC Members: {content_node['note'].strip()}",
                    "metadata": new_metadata
                })
        
        if "events" in content_node and isinstance(content_node["events"], list):
            for event_data in content_node["events"]:
                event_line = (
                    f"Event No. {event_data.get('no', '-')}: {event_data.get('event', '')}, "
                    f"Date: {event_data.get('date', '-')}, Venue: {event_data.get('venue', '-')}"
                )
                if event_line.strip():
                    new_metadata = metadata_prefix.copy()
                    new_metadata["item_type"] = "event"
                    new_metadata["event_no"] = event_data.get('no', '-')
                    new_metadata["event_name"] = event_data.get('event', '')
                    chunks.append({
                        "content": event_line.strip(),
                        "metadata": new_metadata
                    })
        
        if "statements" in content_node and isinstance(content_node["statements"], list):
            for statement_data in content_node["statements"]:
                for key, val in statement_data.items():
                    statement_line = f"{key}: {val}"
                    if statement_line.strip():
                        new_metadata = metadata_prefix.copy()
                        new_metadata["item_type"] = "statement"
                        new_metadata["statement_key"] = key
                        chunks.append({
                            "content": statement_line.strip(),
                            "metadata": new_metadata
                        })

        # Handles the 'Overview' structure specifically, ensuring no duplicate chunks
        # if its content has already been merged into `combined_text_for_this_dict_node`.
        if "Title" in content_node and "Location" in content_node and "Theme and Priorities" in content_node:
            overview_content = (
                f"Title: {content_node['Title']}\n"
                f"Location: {content_node['Location']}\n"
                f"Theme and Priorities: {content_node['Theme and Priorities']}"
            )
            if overview_content.strip():
                if not combined_text_for_this_dict_node or overview_content.strip() not in "\n".join(combined_text_for_this_dict_node):
                    chunks.append({
                        "content": overview_content.strip(),
                        "metadata": metadata_prefix.copy()
                    })

    return chunks

def generate_chunks_from_json(json_data, source_filename="data.json"):
    """
    Main function to traverse the entire JSON structure and generate chunks.
    """
    all_chunks = []
    
    for entry in json_data:
        category = entry.get("category")
        page = entry.get("page")

        for section_obj in entry.get("sections", []):
            section_name = section_obj.get("section")
            content_data = section_obj.get("content")

            initial_metadata = {
                "category": category,
                "page": page,
                "section": section_name,
                "source": source_filename
            }

            # Processes various 'content' types found directly under a section.
            if isinstance(content_data, list):
                for item in content_data:
                    if isinstance(item, dict):
                        if "section" in item and "content" in item:
                            sub_section_name = item["section"]
                            new_metadata = initial_metadata.copy()
                            new_metadata["sub_section"] = sub_section_name
                            sub_chunks = process_content_node(item["content"], new_metadata)
                            all_chunks.extend(sub_chunks)
                        elif "sub_section" in item and "content" in item:
                            sub_section_name = item["sub_section"]
                            new_metadata = initial_metadata.copy()
                            new_metadata["sub_section"] = sub_section_name
                            sub_chunks = process_content_node(item["content"], new_metadata)
                            all_chunks.extend(sub_chunks)
                        elif "Title" in item and "Location" in item:
                            overview_content = (
                                f"Title: {item.get('Title', '')}\n"
                                f"Location: {item.get('Location', '')}\n"
                                f"Theme and Priorities: {item.get('Theme and Priorities', '')}"
                            )
                            if overview_content.strip():
                                all_chunks.append({
                                    "content": overview_content.strip(),
                                    "metadata": initial_metadata.copy()
                                })
            elif isinstance(content_data, dict):
                sub_chunks = process_content_node(content_data, initial_metadata)
                all_chunks.extend(sub_chunks)
            elif isinstance(content_data, str):
                if content_data.strip():
                    all_chunks.append({
                        "content": content_data.strip(),
                        "metadata": initial_metadata.copy()
                    })
    return all_chunks

def remove_duplicate_chunks(chunks):
    """
    Removes duplicate chunks based on their content, ensuring only unique entries remain.
    """
    unique_chunks = []
    seen_content = set()
    for chunk in chunks:
        # Normalizes content by stripping whitespace for consistent comparison.
        normalized_content = chunk["content"].strip()
        if normalized_content and normalized_content not in seen_content:
            unique_chunks.append(chunk)
            seen_content.add(normalized_content)
    return unique_chunks

def assign_chunk_ids(chunks):
    """
    Assigns unique and sequential IDs to chunks after deduplication.
    """
    for i, chunk in enumerate(chunks):
        chunk['id'] = f"chunk_{i}"
    return chunks

if __name__ == "__main__":
    # My specified input file for chunking.
    json_file_name = 'backend\\data\\raw\\apec2025_all_info_20250708_221755.json'
    
    # Defining the output path for processed chunks. I've updated the file name
    # to reflect this refined version.
    output_directory = 'backend/data/processed'
    output_file_name = os.path.join(output_directory, 'refined_processed_chunks_v4.json')

    if not os.path.exists(json_file_name):
        print(f"Error: File '{json_file_name}' not found. Please ensure the file is in the correct directory.")
    else:
        # Load JSON data
        with open(json_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Generate chunks from the loaded data.
        print(f"Generating chunks from file: {json_file_name}...")
        generated_chunks = generate_chunks_from_json(data, json_file_name)
        print(f"Initial total chunks generated (before deduplication and ID assignment): {len(generated_chunks)}")

        # Remove duplicate chunks to maintain data quality.
        print("Removing duplicate chunks...")
        final_chunks_before_id_assign = remove_duplicate_chunks(generated_chunks)
        print(f"Total unique chunks after deduplication: {len(final_chunks_before_id_assign)}")
        
        # Assign unique IDs to the final set of chunks.
        final_chunks = assign_chunk_ids(final_chunks_before_id_assign)

        # Ensure the output directory exists before saving.
        os.makedirs(output_directory, exist_ok=True)
        
        # Save the unique chunks to a new JSON file.
        with open(output_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=4)
        print(f"Unique chunks saved to: {output_file_name}")

        # Providing sample chunks for verification. This helps confirm the chunking
        # logic works as intended for different content types.
        print("\n--- First few chunks after processing ---")
        for i, chunk in enumerate(final_chunks[:10]):
            print(f"\nChunk {i+1} (ID: {chunk['id']})")
            print(f"Content: {chunk['content']}")
            print(f"Metadata: {chunk['metadata']}")
            print("-" * 30)

        print("\n--- Example of separately generated event chunks ---")
        event_chunks = [c for c in final_chunks if c['metadata'].get('item_type') == 'event']
        for i, chunk in enumerate(event_chunks[:5]):
            print(f"\nEvent Chunk {i+1} (ID: {chunk['id']})")
            print(f"Content: {chunk['content']}")
            print(f"Metadata: {chunk['metadata']}")
            print("-" * 30)
        
        print("\n--- Example of combined chunk for 'Gyeongju East Palace Garden' ---")
        garden_chunks = [c for c in final_chunks if c['metadata'].get('sub_section') == 'Gyeongju East Palace Garden']
        if garden_chunks:
            for i, chunk in enumerate(garden_chunks):
                print(f"\nGarden Chunk {i+1} (ID: {chunk['id']})")
                print(f"Content: {chunk['content']}")
                print(f"Metadata: {chunk['metadata']}")
                print("-" * 30)
        else:
            print("No combined chunk found for 'Gyeongju East Palace Garden'.")

        print("\n--- Example of combined chunk for 'Architecture' ---")
        architecture_chunks = [c for c in final_chunks if c['metadata'].get('sub_section') == 'Architecture']
        if architecture_chunks:
            for i, chunk in enumerate(architecture_chunks):
                print(f"\nArchitecture Chunk {i+1} (ID: {chunk['id']})")
                print(f"Content: {chunk['content']}")
                print(f"Metadata: {chunk['metadata']}")
                print("-" * 30)
        else:
            print("No combined chunk found for 'Architecture'.")