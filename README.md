# ai-property-search

🏡 Real Estate Property Search Chatbot:

This is a Streamlit application that acts as an intelligent assistant for searching and summarizing real estate properties. It uses the Gemini API to interpret natural language queries (like "3BHK flats in Pune under 2 Cr ready to move") and translate them into structured filters to search a local property dataset (properties_data_processed.csv).

The chatbot provides a summary of the filtered results, making property hunting efficient and conversational.

✨ Key Features

1.Natural Language Filtering: Use the power of the Gemini API to extract complex search criteria (City, BHK, Price Range, Possession Status, etc.) from plain text.

2.Data-Driven Search: Filters are applied directly to a Pandas DataFrame for fast and accurate results.

3.AI Summary: Generates a concise and informative summary of the matching properties.

4.Interactive UI: Built with Streamlit for a clean, user-friendly, web-based chat interface.

<img width="880" height="445" alt="image" src="https://github.com/user-attachments/assets/531848e1-78a3-420c-add6-97d1683fda56" />


💻 How to Use

1.Start Chatting: Enter your property search query in the text box at the bottom (e.g., "Show me 2 BHK in Mumbai under 1.5 Cr that are ready to move").

2.Submit: Hit Enter or click the send button.

3.Analysis: The app uses the Gemini model to parse the request into structured filters.

4.Results: The app filters the CSV data and uses the Gemini model again to summarize the matching properties.

5.View: The summary and the underlying filters used for the search will be displayed in the chat history.
