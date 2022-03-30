import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text_files=[doc for doc in os.listdir() if doc.endswith('.txt')]
text_note=[open(file,encoding='utf-8').read() for file in text_files]

vectorize=lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity=lambda doc1,doc2: cosine_similarity([doc1,doc2])

vectors=vectorize(text_note)
s_vectors=list(zip(text_files,vectors))

def check_plagiarism():
    plagiarism_results=set()
    global s_vectors
    for textfile_1,textvector_1 in s_vectors:
        new_vectors=s_vectors.copy()
        current_index=new_vectors.index((textfile_1,textvector_1))
        del new_vectors[current_index]
        for textfile_2,textvector_2 in new_vectors:
            sim_score=similarity(textvector_1,textvector_2)[0][1]
            text_pair=sorted((textfile_1,textfile_2))
            score=(text_pair[0],text_pair[1],sim_score*100)
            plagiarism_results.add(score)
    return plagiarism_results

for data in check_plagiarism():
    print(data)
