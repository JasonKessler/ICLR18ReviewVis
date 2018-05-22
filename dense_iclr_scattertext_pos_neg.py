import pandas as pd
import scattertext as st
import spacy

reviews_df = pd.read_csv('https://github.com/JasonKessler/ICLR18ReviewVis/raw/master/iclr2018_reviews.csv.bz2')
reviews_df['parse'] = reviews_df['review'].apply(spacy.load('en'))
corpus = (st.CorpusFromParsedDocuments(reviews_df, category_col = 'rating_bin', parsed_col = 'parse')
          .build().remove_categories(['Neutral']))
html = st.produce_scattertext_explorer(corpus, 
                                       category='Positive', not_categories=['Negative'],
                                       transform = st.Scalers.dense_rank,
                                       term_scorer = st.RankDifference(),
                                       metadata = corpus.get_df()['metadata'])
open('output/pos_neg_dense.html', 'wb').write(html.encode('utf-8'))