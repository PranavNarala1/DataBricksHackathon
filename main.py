from flask import Flask,render_template,request
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

import pinecone

import pandas as pd

pinecone.init(api_key="---", environment="us-west4-gcp-free")

pinecone_index = pinecone.Index("quickstart")

print(pinecone_index)

publications = pd.read_csv('static/files/final_data.csv')

app = Flask(__name__)


@app.route('/',  methods=["GET", "POST"])
def index():
  if request.method == "POST":
    form_data = request.form
    user_input = str(form_data["Name"])

    embedding = model.encode([user_input])
    input_vec = [float(x) for x in embedding[0]]

    search_results = [pinecone_index.query(vector= [float(x) for x in embedding[0]],top_k=10,include_values=True)['matches'][x]['id'] for x in range(10)]
    #print(search_results)

    arxiv_links = []

    for x in range(5):
      proj_title = ' '.join(search_results[x].split()[2:])

      #print(publications[publications['Title'] == proj_title])
      #print(publications[publications['Title'] == proj_title])
      #print(publications[publications['Title'] == proj_title]['URL'])
      arxiv_links.append(list(publications[publications['Title'] == proj_title]['URL'])[0])
    


    return render_template("matches.html", search_matches=search_results, arxiv_links=arxiv_links)

    #get text data
    #run database code
    #get results and pass it into matches
  #  matches(search_matches)
  return render_template("index.html")

@app.route('/matches')
def matches(search_matches, arxiv_links):
  return render_template("matches.html", search_matches=search_matches, arxiv_links=arxiv_links)


app.run(host='0.0.0.0', port=130)
