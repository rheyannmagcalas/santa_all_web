from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
from rest_framework import generics

import json
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import os
import json
import pickle
import re
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


import json
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import os
import json
import pickle
import re
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def get_tfidf(product_details):
    clean_product = []
    product_name = list(product_details)
    for i in range(len(product_name)):
        words = ""

        doc = nlp(product_name[i].lower())
        for token in doc:
            token.lemma_ = re.sub(r'\W',' ',token.lemma_)
            token.lemma_ = token.lemma_.strip()
            if not token.lemma_.endswith("ml") and not token.lemma_.endswith("ms") and not token.lemma_.isdigit() and not token.lemma_ in stop_words:
                if len(token.lemma_) > 2 or token.lemma_ == 'uv': 
                    words += token.lemma_.lower() + " "
                    

        if len(words) > 0:
            clean_product.append(str(words.strip()))

    tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(clean_product)
    first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), 
                      columns=["tfidf"]) 
    df = df.sort_values(by=["tfidf"], ascending=False).reset_index()
    
    return df

def index(request):
    return render(request, 'wishlist/index.html')

def register(request):
    return render(request, 'registration/register.html')

def forgot_password(request):
    return render(request, 'registration/forgot_password.html')

def mywishlist(request):
    return render(request, 'wishlist/my_wishlist.html')

def community(request):
    return render(request, 'wishlist/community.html')    

class SearchWishlist(generics.ListAPIView):


    def get(self, request):
        result = []
        product_name = self.request.query_params.get('product-name', '')

        with open('/Users/rhey.magcalas/codes/research/secret_santa_web/main/network_theory.pickle','rb') as fe_data_file:
            G = pickle.load(fe_data_file)

        with open('/Users/rhey.magcalas/codes/research/secret_santa_web/main/betweenness_centraility.json') as f:
            between_centrality_json = json.load(f)

        overall_data = pd.read_csv('/Users/rhey.magcalas/codes/research/secret_santa_web/main/cleaned_data.csv', error_bad_lines=False)
        
        sub_category_list_2 = list(overall_data['Sub Category 2'].str.lower().unique())

        doc = nlp(product_name.strip().lower())
        result_categories = []

        for token in reversed(doc):
            if token.text in list(G.nodes()):
                print(token.lemma_)
                closeness_centrality_list = []
                betweness_centrality_list = []
                degree_list = []
                neighbor_list = []
                shortest_path_list = []
                length_list = []

                for _neighbors in list(G.neighbors(token.text)):
                    if _neighbors in sub_category_list_2:
                        neighbor_list.append(_neighbors)
                        betweness_centrality_list.append(between_centrality_json[_neighbors])
                        shortest_path = nx.shortest_path(G, source=_neighbors, target=token.lemma_)
                        shortest_path_list.append(len(shortest_path))
                        length_list.append(overall_data.loc[overall_data['Sub Category 2'] == _neighbors].shape[0])

                network_result = pd.DataFrame(neighbor_list, columns=['neighbor'])
                network_result['betweeness_centrality'] = betweness_centrality_list
                network_result['shortest_path'] = shortest_path_list

                if len(betweness_centrality_list) > 0:
                    if network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'].shape[0] < 2:
                        if list(network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'])[0] not in result_categories:
                            result_categories.append(list(network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'])[0])
                    else:
                        if list(network_result[network_result['betweeness_centrality'] == min(betweness_centrality_list)]['neighbor'])[0] not in result_categories:
                            result_categories.append(list(network_result[network_result['betweeness_centrality'] == min(betweness_centrality_list)]['neighbor'])[0]) 
        merge_products = []
        for _result_categories in result_categories:
            merge_products.append(overall_data.loc[(overall_data['Sub Category 2'] == _result_categories.title())])

        selected_category = pd.concat(merge_products).reset_index()

        vectorize = TfidfVectorizer(stop_words='english')
        tfidf_response= vectorize.fit_transform(selected_category['Product Name'])
        dtm = pd.DataFrame(tfidf_response.todense(), columns = vectorize.get_feature_names())

        nn = NearestNeighbors(n_neighbors=selected_category.shape[0])
        nn.fit(dtm)
        wishlist = [product_name.strip().lower()]

        new = vectorize.transform(wishlist)
        knn_model_result = nn.kneighbors(new.todense())

        knn_result = pd.DataFrame(knn_model_result[0][0][0:], columns=['Distance'])
        knn_result["Product Name"] = selected_category['Product Name'][knn_model_result[1][0][0:]]

        merged_result = pd.merge(selected_category, knn_result, on='Product Name', how='inner')
        merged_result = merged_result.drop_duplicates(subset='Product Name', keep="first")

        scaler = MinMaxScaler()

        scoring_criteria = ['Trusted', 'Highly Rated', 'Discounted', 'Top Selling', 'High Interest']
        df_eng = merged_result.copy()
        df_eng['Discount Percent'] = df_eng['Discount Range']-df_eng['Price Range']
        df_eng['Total Sold'] = scaler.fit_transform(df_eng[['Total Sold']])
        df_eng['High Interest'] = scaler.fit_transform(df_eng[['Favorite']])

        # Conditions
        df_eng['Highly Rated'] = df_eng['Current Rating'].astype(float).map(lambda x: True if x>4.2 else False)
        df_eng['Discounted'] = df_eng['Discount Percent'].astype(float).map(lambda x: True if x>0.03 else False)
        df_eng['Top Selling'] = df_eng['Total Sold'].map(lambda x: True if x>0.15 else False)
        df_eng['High Interest'] = df_eng['High Interest'].map(lambda x: True if x>0.15 else False)

        # New Columns
        df_eng['Trusted'] = df_eng.apply(lambda x: x['Preferred'] | x['Mall'], axis=1)

        model_features = ['Price Range']
        scoring_criteria = ['Trusted', 'Highly Rated', 'Discounted', 'Top Selling', 'High Interest']
        if df_eng.shape[0] < 10:
            prd_list = df_eng.sample(n=df_eng.shape[0])
        else:
            prd_list = df_eng.head(50)
        prd_list['relevance'] = np.random.uniform(0, 1, prd_list.shape[0])
        prd_list['score'] = prd_list['relevance']

        scored_list = prd_list[prd_list['Current Rating'] > 3.8]

        # Scoring System
        trusted_bias = 0.05
        highly_rated_bias = 0.05
        discounted_bias = 0.05
        top_selling_bias = 0.05
        high_interest_bias = 0.05

        scored_list['score'] = scored_list.apply(lambda x: x['score']+trusted_bias if x['Trusted'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']+highly_rated_bias if x['Highly Rated'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']+discounted_bias if x['Discounted'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']+top_selling_bias if x['Top Selling'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']+high_interest_bias if x['High Interest'] == True else x['score'], axis=1)
        scored_list

        for index, row  in scored_list.iterrows():
            current_data = overall_data[overall_data['Product ID'] == row['Product ID']]
            # print(current_data['URL'].iloc[0])

            image_url = ''
            if 0 in current_data['Image URL'].index:
                image_url = current_data['Image URL'].iloc[0]

            result.append({
                'url': current_data['URL'].iloc[0],
                'name': current_data['Product Name'].iloc[0],
                'image_url': image_url,
                'rating': str(current_data['Current Rating'].iloc[0]),
                'price': str(current_data['Price Range'].iloc[0]),
                'total_sold': str(current_data['Total Sold'].iloc[0]),
                'total_rating': str(current_data['Total Rating'].iloc[0]),
                'free_shipping': str(current_data['Free Shipping Info'].iloc[0])
            })

        return JsonResponse(result, safe=False)

