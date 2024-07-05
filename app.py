from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# loading cached recommendations and articles
recs_cache = np.load('top_20_recommendations.npy')
df = pd.read_csv('articles_full.csv')

# getting recommendations
def get_recommendations_by_title(title, top_n=5):
    try:
        id = df[df['title'] == title].index[0]
    except:
        return None, None
    
    original_article = df.iloc[id]
    rec_indices = recs_cache[id][:top_n]
    return original_article, df.iloc[rec_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recs = None
    title = None
    top_n = 5
    original_article = None

    if request.method == 'POST':
        title = request.form['title']
        top_n = int(request.form['top_n'])
        original_article, recs = get_recommendations_by_title(title, top_n)
    
    return render_template('index.html', title=title, top_n=top_n,
                           original_article=original_article.to_dict() if original_article is not None else None,
                           recommendations=recs.to_dict(orient='records') if recs is not None else None)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '')
    matches = df[df['title'].str.contains(query, case=False, na=False)].head(5)['title'].tolist()
    return jsonify(matches)

@app.route('/search_articles', methods=['GET'])
def search_articles():
    title = request.args.get('title', '')
    keyword = request.args.get('keyword', '')
    date = request.args.get('date', '')
    page = int(request.args.get('page', 1))
    page_size = 10

    filtered_df = df
    if title:
        filtered_df = filtered_df[filtered_df['title'].str.contains(title, case=False, na=False)]
    if keyword:
        filtered_df = filtered_df[filtered_df['summary'].str.contains(keyword, case=False, na=False)]
    if date:
        filtered_df = filtered_df[filtered_df['date'] == date]

    total_articles = len(filtered_df)
    total_pages = (total_articles + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size

    articles = filtered_df.iloc[start:end].to_dict(orient='records')
    return jsonify({
        'articles': articles,
        'total_pages': total_pages,
        'current_page': page
    })

if __name__ == '__main__':
    app.run(debug=True)
