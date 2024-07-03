from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# loading cached recommendations and articles
recs_cache = np.load('top_20_recommendations.npy')
df = pd.read_csv('articles_full.csv')

# getting recommendations
def get_recommendations_by_title(title, top_n = 5):
    try:
        id = df[df['title'] == title].index[0]
    except:
        return None, None
    
    original_article = df.iloc[id]

    rec_indices = recs_cache[id][ : top_n]
    return original_article, df.iloc[rec_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recs = None
    title = ""
    original_article = None
    top_n = ""

    if request.method == 'POST':
        title = request.form['title']
        top_n = int(request.form['top_n'])
        original_article, recs = get_recommendations_by_title(title, top_n)
    
    return render_template('index.html', title = title, top_n = top_n,
                           original_article = original_article.to_dict() if original_article is not None else None,
                           recommendations= recs.to_dict(orient = 'records') if recs is not None else None)

# for autocompleting titles
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '')
    matches = df[df['title'].str.contains(query, case = False, na = False)].head(5)['title'].tolist()
    return jsonify(matches)

if __name__ == '__main__':
    app.run(debug = True)