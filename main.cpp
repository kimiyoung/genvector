
#include "data.hpp"
#include "model.hpp"
#include "config.hpp"

#include <utility>
#include <algorithm>

using namespace std;

int main() {
    int D, W; // D: number of documents (authors), W: number of words
    document * docs; // document arrays
    float ** f_r, ** f_k; // f_r: author embeddings, f_k: keyword embeddings

    read_data(D, W, docs, f_r, f_k); // read the data

    model m(docs, D, W, f_r, f_k); // initialize the model

    m.learn(); // training

    m.save_model(SAVED_MODEL_FILE); // save model to file
}
