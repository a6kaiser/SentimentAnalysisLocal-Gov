import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
import time
from typing import List, Tuple
import sqlite3
import openai
import json

def scrape_council_members(county: str, state: str) -> List[str]:
    """
    Scrape council member names for a given county and state using GPT-4 to parse the content.
    """
    search_query = f"{county} {state} council members"
    council_members = set()
    
    print(f"\nSearching Google for: {search_query}")
    
    try:
        print("Starting Google search...")
        url_count = 0
        combined_text = ""
        
        for url in search(search_query, num_results=3):
            if url_count >= 3:
                break
                
            url_count += 1
            print(f"\nProcessing URL {url_count}/3: {url}")
            
            try:
                print("Fetching webpage...")
                response = requests.get(url, timeout=10)
                print(f"Response status code: {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                # Get text content from main content areas
                main_content = soup.find_all(['main', 'article', 'div'], 
                    class_=re.compile(r'content|main|council|members|officials', re.I))
                
                for content in main_content:
                    combined_text += content.get_text() + "\n"
                
                time.sleep(2)  # Respect rate limits
                
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        
        if combined_text:
            print("\nAsking GPT to extract council members...")
            
            # Configure OpenAI
            client = openai.OpenAI()  # Updated initialization
            
            prompt = f"""
            Extract council members and their positions from the following text about {county}, {state}.
            Return the result as a JSON array of objects with 'name' and 'position' fields.
            Only include current council members, commissioners, or supervisors.
            If you're not confident about a name or position, don't include it.
            
            Text:
            {combined_text[:4000]}  # Limiting text length to avoid token limits
            """
            
            response = client.chat.completions.create(  # Updated API call
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts council member information and returns it in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            try:
                members_data = json.loads(response.choices[0].message.content)
                for member in members_data:
                    name = member.get('name')
                    position = member.get('position')
                    if name and position:
                        council_members.add(f"{name} ({position})")
                        print(f"Found: {name} - {position}")
            except json.JSONDecodeError as e:
                print(f"Error parsing GPT response: {e}")
                print("Raw response:", response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
    
    return list(council_members)

def get_locations() -> List[Tuple[str, str]]:
    """
    Get unique county-state pairs from the meetings database.
    
    Returns:
        List of tuples containing (location_name, location_state)
    """
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT location_name, location_state 
        FROM meetings 
        WHERE location_name IS NOT NULL 
        AND location_state IS NOT NULL
    ''')
    
    locations = cursor.fetchall()
    conn.close()
    return locations

def init_council_members_db():
    """Initialize the council members database table"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS council_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT,
            location_state TEXT,
            member_name TEXT,
            position TEXT,
            UNIQUE(location_name, location_state, member_name)
        )
    ''')
    
    conn.commit()
    conn.close()

def store_council_members(location_name: str, location_state: str, members: List[str]):
    """Store council members in the database"""
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    
    for member in members:
        # Parse the member string which is in format "name (position)"
        name, position = member.split('(', 1)
        name = name.strip()
        position = position.rstrip(')')
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO council_members 
                (location_name, location_state, member_name, position)
                VALUES (?, ?, ?, ?)
            ''', (location_name, location_state, name, position))
            print(f"Stored: {name} - {position}")
        except sqlite3.Error as e:
            print(f"Error storing {name}: {e}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Initialize the database
    init_council_members_db()
    
    # Get all unique locations from database
    locations = get_locations()
    print(f"Found {len(locations)} unique locations")
    
    # Process each location
    for location_name, location_state in locations:
        print(f"\nProcessing: {location_name}, {location_state}")
        members = scrape_council_members(location_name, location_state)
        print(f"Found council members:")
        for member in members:
            print(f"- {member}")
        
        # Store the members in the database
        store_council_members(location_name, location_state, members)
