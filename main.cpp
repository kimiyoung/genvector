
#include "data.hpp"
#include "model.hpp"
#include "config.hpp"

#include <utility>
#include <algorithm>

using namespace std;

bool comp(const pair<int, float> p1, const pair<int, float> p2) {
    return p1.second > p2.second;
}

char temp_[50];

int main() {
    int D, W;
    document * docs;
    float ** f_r, ** f_k;

    read_data(D, W, docs, f_r, f_k);

    model m(docs, D, W, f_r, f_k);

    FILE * fin, * fout;
    fin = fopen(KEYWORD_INDEX, "r");

    char ** keyword = new char * [W];
    for (int i = 0; i < W; i ++) {
        keyword[i] = new char[50];
        fscanf(fin, "%s", keyword[i]);
    }
    fclose(fin);

    m.sample_topics();
    m.save_model(SAVED_MODEL_FILE);
    for (int t = 0; t < 3; t ++) {
        m.embedding_update();
        m.sample_topics();
        m.embedding_update();
        m.sample_topics();
        m.embedding_update();
        m.sample_topics();
    }

    m.save_model(SAVED_MODEL_FILE);
}
