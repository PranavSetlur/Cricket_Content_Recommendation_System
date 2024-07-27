# Cricket Article Recommendations System

This system provides recommendations for cricket-related articles based on user input and allows searching through a comprehensive database of articles. You can find the app here - https://psetlur6.pythonanywhere.com/. Read more about the app here - https://between22yards.wordpress.com/2024/07/27/cricket-content-curator/.

## Setup Instructions

### Prerequisites

- Python (version 3.6 or higher)
- Flask framework
- Pandas library
- numpy library

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/PranavSetlur/Cricket_Content_Recommendation_System.git
   cd Cricket_Content_Recoomendation_System
   ```
2. **Install Python dependencies:**

```
pip install -r requirements.txt
```
3. **Run the Application**
   ```
   python app.py
   ```
This command starts the Flask development server. By default, the application runs on http://localhost:5000.

## Usage
Access the Application:

Open a web browser and go to http://localhost:5000 to access the application.

### Recommendations:

1. Enter an article title in the input field on the "Recommendations" tab.
2. Specify the number of recommendations you wish to retrieve.
3. Click on the "Get Recommendations" button to see a list of recommended articles.

### Article Database
1. Navigate to the "Article Database" tab.
2. Use the search fields to filter articles by title and summary
3. The articles matching your search criteria will be displayed in a table format with columns for the title, summary, and published daye
4. Use the paginated controls at the bottom to navigate through the pages of articles.

You can also use the article database tab to find the title of a specific article you want recommended for you.
