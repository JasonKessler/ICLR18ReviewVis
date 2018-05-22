import pandas as pd
import scattertext as st

reviews_df = pd.read_csv('https://github.com/JasonKessler/ICLR18ReviewVis/raw/master/iclr2018_reviews.csv.bz2')
reviews_df['parse'] = reviews_df['review'].apply(st.whitespace_nlp_with_sentences)
corpus = (st.CorpusFromParsedDocuments(reviews_df, category_col = 'decision', parsed_col = 'parse')
          .build().remove_categories(['Workshop']))
html = st.produce_scattertext_explorer(corpus, 
                                       category='Accept', not_categories=['Reject'],
                                       transform = st.Scalers.dense_rank,
                                       term_scorer = st.RankDifference(),
                                       metadata = corpus.get_df()['metadata'])
open('output/accept_reject_dense.html', 'wb').write(html.encode('utf-8'))
