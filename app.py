import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 0. Setup and Initialization ---

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        # Initialize client
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Gemini Client: {e}")
        client = None
else:
    # This error will show up if .env is missing or key is empty
    st.error("GEMINI_API_KEY not found. Please set it in the .env file.")
    client = None

# Constants
DATA_FILE = 'properties_data_processed.csv'
MODEL = 'gemini-2.5-flash'

# JSON Schema for Filter Extraction (Explicitly defined to prevent parsing errors)
FILTER_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "city": types.Schema(type=types.Type.STRING, description="The city, e.g., 'Pune' or 'ANY' or 'Mumbai, Pune'"),
        "bhk": types.Schema(type=types.Type.STRING, description="The BHK configuration, e.g., '3BHK', '1RK', or 'ANY'"),
        "min_price": types.Schema(type=types.Type.NUMBER, description="The minimum price in Indian Rupees (e.g., 0, 5000000)."),
        "max_price": types.Schema(type=types.Type.NUMBER, description="The maximum price in Indian Rupees (e.g., 12000000). Use -1 if no upper limit is specified."),
        "possession_status": types.Schema(type=types.Type.STRING, description="The possession status: 'READY_TO_MOVE', 'UNDER_CONSTRUCTION', or 'ANY'")
    },
    required=["city", "bhk", "min_price", "max_price", "possession_status"]
)


# System Prompt for Filter Extraction (Structured Output)
EXTRACTION_PROMPT = """
You are an expert real estate natural language parser. Your task is to extract key property filters from a user query and map them to the provided JSON schema.

Conversion Rules:
1. Budget: Convert all budget mentions (e.g., '1.2 Cr', '1 Crore', '50 Lakh') into a single number representing Indian Rupees (e.g., 1.2 Cr -> 12000000.0, 50 Lakh -> 5000000.0).
2. Price Range: If the query specifies 'under X', set 'max_price' to X and 'min_price' to 0. If it specifies 'between X and Y', set 'min_price' to X and 'max_price' to Y.
3. Defaults:
    - If a price is not specified, set 'max_price' to -1 (indicating no upper limit) and 'min_price' to 0.
    - If 'possession_status' is not clear, use 'ANY'.
    - If 'city' or 'bhk' is missing, set it to 'ANY'.
"""

# System Prompt for Summarization
SUMMARIZATION_PROMPT = """
You are a helpful real estate assistant. Your task is to generate a short, 4-5 sentence summary for a list of property search results.
The summary MUST be grounded ONLY in the data provided below and MUST NOT hallucinate or use external information.
Focus on:
1. The total number of results found.
2. The range of prices and carpet areas in the results.
3. The possession status (Ready-to-move or Under Construction) of the properties.
4. The most common locality/city.

Total Properties Found: {total_count}

Data (Sample of properties):
---
{data_summary}
---
"""

# --- 1. Data Loading and Caching ---

@st.cache_data
def load_data():
    """Loads and preprocesses the property data."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Error: Data file '{DATA_FILE}' not found. Ensure you have the file in the project directory.")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_FILE)
    
    # Ensure price and carpetArea are numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['carpetArea'] = pd.to_numeric(df['carpetArea'], errors='coerce')

    # Convert to Cr/Lakh for display later
    df['price_display'] = df['price'].apply(format_price)
    
    # Check if essential columns are present
    required_cols = ['price', 'bhkType', 'city', 'projectName', 'possessionStatus', 'fullAddress', 'slug']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Error: Missing required column '{col}' in '{DATA_FILE}'. Please check your CSV file.")
            return pd.DataFrame()
            
    return df.dropna(subset=['price', 'bhkType'])

# --- 2. Helper Functions ---

def format_price(price):
    """Formats price in Lakhs (L) or Crores (Cr)."""
    if pd.isna(price):
        return "Price on Request"
    price = float(price)
    if price >= 10000000:
        return f"₹{price / 10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹{price / 100000:.2f} L"
    else:
        return f"₹{price:,.0f}"

def format_bhk(bhk_type, custom_bhk):
    """Combines bhkType and customBHK for a cleaner title."""
    if pd.notna(custom_bhk) and str(custom_bhk) not in ['None', 'NaN']:
        return custom_bhk
    return bhk_type
    
# --- 3. Core LLM Functions ---

def extract_filters(query: str):
    """
    Uses Gemini to extract structured filters from the natural language query.
    Uses the explicitly defined FILTER_SCHEMA.
    """
    if not client:
        return None

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[EXTRACTION_PROMPT + "\n\nUser Query: " + query],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FILTER_SCHEMA # Direct use of the defined schema
            )
        )
        # Assuming the response text is a valid JSON string matching the schema
        return json.loads(response.text)
    except Exception as e:
        # Catch any API or JSON parsing errors
        st.error(f"LLM Error (Filter Extraction): {e}")
        return None

def generate_summary(results_df: pd.DataFrame, filters: dict):
    """
    Uses Gemini to generate a summary based on the filtered results.
    """
    if not client:
        return "Cannot generate summary: LLM client not initialized."

    if results_df.empty:
        # Graceful fallback logic
        city = filters.get('city', 'the specified area')
        bhk = filters.get('bhk', 'property')
        max_price_display = format_price(filters.get('max_price', 0))
        
        fallback_msg = f"No {bhk} options found under {max_price_display} in {city}. Please try expanding your budget or searching for a different configuration."
        return fallback_msg

    # Prepare data summary for the LLM (sample up to 10 rows for grounding)
    data_points = results_df[['projectName', 'city', 'bhkType', 'price_display', 'carpetArea', 'possessionStatus']].head(10).to_string(index=False)
    
    # Pass the total count explicitly for accurate reporting
    total_count = len(results_df)
    
    prompt = SUMMARIZATION_PROMPT.format(data_summary=data_points, total_count=total_count)
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
        )
        return response.text.strip()
    except Exception as e:
        return f"LLM Error (Summarization): Could not generate a summary due to an API issue: {e}"

# --- 4. Search and Retrieval ---

def search_properties(df: pd.DataFrame, filters: dict):
    """
    Filters the DataFrame based on the extracted criteria.
    Handles multiple cities (e.g., 'Mumbai or Pune' or 'Mumbai, Pune').
    """
    filtered_df = df.copy()
    
    # 1. City Filter
    city_filter_val = filters.get('city')
    
    if city_filter_val and city_filter_val.upper() != 'ANY':
        # Split by comma, ' or ', or just spaces, then strip whitespace and convert to uppercase
        city_list = [
            c.strip().upper() for c in 
            city_filter_val.replace(' or ', ',').replace(' OR ', ',').split(',') 
            if c.strip()
        ]
        
        if city_list:
            # Filter the DataFrame where the city column value is in the parsed list of cities
            filtered_df = filtered_df[filtered_df['city'].str.upper().isin(city_list)]

    # 2. BHK Filter
    bhk = filters.get('bhk')
    if bhk and bhk.upper() != 'ANY':
        # Match both bhkType and customBHK columns
        filtered_df = filtered_df[
            (filtered_df['bhkType'].str.upper() == bhk.upper()) |
            (filtered_df['customBHK'].astype(str).str.upper() == bhk.upper())
        ]

    # 3. Budget Filter (Min/Max Price)
    min_price = filters.get('min_price', 0)
    max_price = filters.get('max_price', -1)
    
    if min_price > 0:
        filtered_df = filtered_df[filtered_df['price'] >= min_price]
    if max_price > 0:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]

    # 4. Possession Status Filter
    status = filters.get('possession_status')
    if status and status.upper() != 'ANY':
        # Filter on exact match of possessionStatus
        filtered_df = filtered_df[filtered_df['possessionStatus'].str.upper() == status.upper()]
        
    return filtered_df.sort_values(by='price', ascending=True)

# --- 5. Streamlit UI (Frontend) ---

def create_property_card(row, index):
    """
    Generates an interactive and visually improved card for a single property result.
    """
    
    # Placeholder for amenities (replace with real data if available)
    amenities = ["Clubhouse", "Lift", "Power Backup"]

    # Safely get possession status and format color
    status = row.get('possessionStatus', 'UNKNOWN').upper()
    status_display = status.replace('_', ' ').title()

    # Determine status icon and color
    if status == 'READY_TO_MOVE':
        status_icon = "✅"
        status_color = "linear-gradient(90deg, #10B981, #059669)" 
        status_text_color = "white"
    elif status == 'UNDER_CONSTRUCTION':
        status_icon = "🏗️"
        status_color = "linear-gradient(90deg, #F59E0B, #D97706)"
        status_text_color = "black"
    else:
        status_icon = "❓"
        status_color = "#374151" 
        status_text_color = "white"


    # --- HTML/Markdown Styling for a Dark, Attractive Card ---
    
    # Custom CSS for component styling
    card_style = """
    <style>
    .card-title {
        font-size: 1.8em; 
        font-weight: 800; 
        color: #FACC15; 
        margin-bottom: 0px;
    }
    .card-subtitle {
        font-size: 1.1em;
        font-weight: 600;
        color: #D1D5DB; 
        margin-top: 0px;
        margin-bottom: 5px;
    }
    .info-tag {
        font-size: 1.2em; 
        font-weight: 700;
        color: #E5E7EB; 
        display: block; 
        padding-bottom: 5px;
    }
    .amenities-list {
        font-size: 0.9em;
        color: #9CA3AF; 
        font-style: italic;
    }
    </style>
    """

    # Create the card visually using a native container with a border
    with st.container(border=True):
        
        # --- 1. Project Name (Title) ---
        st.markdown(card_style, unsafe_allow_html=True) 
        st.markdown(f"<p class='card-title'>{row['projectName']}</p>", unsafe_allow_html=True)
        
        st.markdown("---") # Visual separator

        # --- 2. Key Metrics (BHK and Price) ---
        col_bhk, col_price = st.columns([1, 1], gap="small")
        
        with col_bhk:
            bhk_text = format_bhk(row['bhkType'], row.get('customBHK', ''))
            # Removed ** from the value
            st.markdown(f"<div class='info-tag'>BHK: <span style='color: #38BDF8;'>{bhk_text}</span></div>", unsafe_allow_html=True) 
        
        with col_price:
            # Removed ** from the value
            st.markdown(f"<div class='info-tag'>Price: <span style='color: #FACC15;'>{row['price_display']}</span></div>", unsafe_allow_html=True) 

        st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
        
        # --- 3. Locality and Address ---
        # FIXED: Removed ** from around the city name and wrapped the city in a <span> tag for bold styling
        city_display = f"<span style='font-weight: 900;'>{row['city'].capitalize()}</span>"
        address_part = row['fullAddress'].split(',')[0]
        st.markdown(f"<p class='card-subtitle'>Locality: {city_display} | {address_part}</p>", unsafe_allow_html=True)
        
        # --- 4. Possession Status (Prominent Button/Badge Look) ---
        st.markdown(
            f"""
            <div style="background: {status_color}; padding: 10px; border-radius: 8px; text-align: center; margin-top: 15px; margin-bottom: 10px;">
                <span style="font-size: 1.1em; font-weight: 700; color: {status_text_color};">
                    {status_icon} {status_display}
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
        # --- 5. Amenities ---
        # Removed ** around the amenities list value
        st.markdown(f"<p class='amenities-list'>Top Amenities: {', '.join(amenities)}</p>", unsafe_allow_html=True)
        
        # --- 6. Details Expander ---
        with st.expander(label="View Details", expanded=False):
            st.markdown("---")
            st.markdown(f"**Project Name:** `{row['projectName']}`")
            st.markdown(f"**Full Address:** `{row['fullAddress']}`")
            st.markdown(f"**Carpet Area:** `{row.get('carpetArea', 'N/A')} sqft`")
            st.markdown(f"**Possession Status:** `{status_display}`")
            
            # Detailed table view
            details_df = row.to_frame('Value').reset_index()
            details_df.columns = ['Field', 'Value']
            details_df = details_df[~details_df['Field'].isin(['price', 'slug', 'customBHK', 'price_display'])]
            
            st.dataframe(details_df, use_container_width=True, hide_index=True)

    return "" 


# Card display logic now uses a consistent grid
def display_property_grid(top_results: pd.DataFrame):
    """
    Displays property cards in a consistent 3-column grid with equal gaps.
    """
    # Define the number of columns per row
    COLUMNS_PER_ROW = 3 
    
    # Iterate over the results in chunks of 3
    for i in range(0, len(top_results), COLUMNS_PER_ROW):
        # Create a new row of columns with equal width and a medium gap
        cols = st.columns(COLUMNS_PER_ROW, gap="medium") 
        
        # Iterate over the columns in the current row
        for j in range(COLUMNS_PER_ROW):
            # Check if there is a property for this column index
            if i + j < len(top_results):
                property_data = top_results.iloc[i + j]
                
                # Use the 'with' syntax to place the card content into the column
                with cols[j]:
                    create_property_card(property_data, i + j)


def main():
    # Set a more engaging page config
    st.set_page_config(
        page_title="AI Property Search", 
        layout="wide", 
        initial_sidebar_state="collapsed", 
        page_icon="🏠"
    )

    st.header("🏠 NoBrokerage.com AI Property Search", divider='blue')
    st.caption("Ask for properties using natural language, e.g., **'3BHK flat in Pune under ₹1.2 Cr ready-to-move'**")

    # Load Data (cached)
    data_df = load_data()
    if data_df.empty:
        # Stop execution if data loading failed or required columns are missing
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI property finder. What kind of property are you looking for?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"]) 
            
            if "filters" in message:
                # Use a small expander to show the filters, keeping the main content clean
                with st.expander("🔍 Show Extracted Filters"):
                    f = message["filters"]
                    # Display Extracted Filters for debugging/transparency
                    st.markdown(f"**BHK:** `{f.get('bhk', 'ANY')}` | **City:** `{f.get('city', 'ANY')}`")
                    st.markdown(f"**Price Range:** `{format_price(f.get('min_price', 0))}` - `{format_price(f.get('max_price', -1))}`")
                    st.markdown(f"**Status:** `{f.get('possession_status', 'ANY')}`")


            if "results" in message and not message["results"].empty:
                st.subheader(f"Showing Top {len(message['results'])} Results")
                
                top_results = message['results']
                
                # Display the grid of property cards
                display_property_grid(top_results)


    # Accept user input
    if prompt := st.chat_input("Enter your property search query..."):
        
        # --- START OF FIX: Immediately store and display the user's message ---
        
        # 1. Immediately store the user message in history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # 2. Immediately display the user message in the current chat run
        with st.chat_message("user"):
            st.write(prompt)
            
        # --- END OF FIX: User message is now secured and displayed ---

        # Start AI processing
        with st.chat_message("assistant"):
            
            # Use an empty placeholder to initially show "Processing..." and later replace it with the summary
            assistant_placeholder = st.empty()
            assistant_placeholder.write("Processing your request...") 
            
            with st.spinner("Analyzing query and searching data..."):
                
                # Step 1: Query Understanding (LLM)
                filters = extract_filters(prompt)
                
                if not filters:
                    error_msg = "Could not parse your query. Please rephrase or ensure your API key is valid."
                    assistant_placeholder.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    return

                # Show Extracted Filters in the current chat bubble for immediate context
                with st.expander("🔍 Extracted Filters (for search)", expanded=True):
                    st.markdown(f"**BHK:** `{filters.get('bhk', 'ANY')}` | **City:** `{filters.get('city', 'ANY')}`")
                    st.markdown(f"**Price Range:** `{format_price(filters.get('min_price', 0))}` - `{format_price(filters.get('max_price', -1))}`")
                    st.markdown(f"**Status:** `{filters.get('possession_status', 'ANY')}`")

                # Step 2: Search & Retrieval (Pandas)
                results_df = search_properties(data_df, filters)
                
                # Step 3: Summarization Logic (LLM)
                summary_text = generate_summary(results_df, filters)
                
                # Update the assistant's content using the placeholder
                assistant_placeholder.write(summary_text)
                
                # Step 4: Prepare and Append Final Assistant Message to history
                if not results_df.empty:
                    top_results = results_df.copy()
                    
                    new_message = {
                        "role": "assistant", 
                        "content": summary_text, 
                        "results": top_results,
                        "filters": filters
                    }
                else:
                    new_message = {"role": "assistant", "content": summary_text, "filters": filters}
                
                # Append the final message to session state. This will be displayed on rerun.
                st.session_state.messages.append(new_message)
                
                # Rerun the script to update the full history display
                st.rerun()


if __name__ == "__main__":
    main()